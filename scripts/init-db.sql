CREATE EXTENSION IF NOT EXISTS timescaledb;
  -- TimescaleDB: Turns PostgreSQL into a time-series database
  -- Why: Financial data is time-based (transactions have timestamps)
  -- Benefit: 10-100x faster queries on time ranges, automatic partitioning
CREATE EXTENSION IF NOT EXISTS pgcrypto;
  -- pgcrypto: Cryptographic functions
  -- Why: We need gen_random_uuid() for generating unique IDs
  -- Alternative: Use Python's uuid4(), but DB-generated is faster

-- EVENT STORE (Write Side)
-- the SINGLE SOURCE OF TRUTH for all state changes
CREATE TABLE events (
      -- Unique identifier for this specific event
      -- UUID prevents collisions across distributed systems
      event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       -- What TYPE of thing does this event belong to?
      -- Examples: 'account', 'transaction', 'budget', 'user'
      aggregate_type VARCHAR(50) NOT NULL,
      -- WHICH specific account/transaction/budget?
      -- This groups all events for one entity together
      aggregate_id UUID NOT NULL,
       -- What happened? (past tense!)
      -- Examples: 'TransactionCreated', 'BudgetExceeded', 'AccountOpened'
      event_type VARCHAR(100) NOT NULL,
       -- Extra info: who triggered it, IP address, request ID
      -- Useful for debugging and audit trails
      event_data JSONB NOT NULL,
      metadata JSONB DEFAULT '{}',
      -- VERSION NUMBER - Critical for optimistic concurrency!
      -- Increments with each event for this aggregate
      -- Version 1, 2, 3, 4... for each aggregate_id
      version INTEGER NOT NULL,
      -- When was this event recorded?
      -- TIMESTAMPTZ = timestamp with timezone (always stores UTC)
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        -- UNIQUE CONSTRAINT: Prevents two events with same version
      -- This is how we detect concurrent modifications!
      UNIQUE (aggregate_id, version)
  );
  -- TIMESCALEDB HYPERTABLE
  -- Automatically partitions data by time for massive performance gains
    SELECT create_hypertable('events', 'created_at');
      -- Convert 'events' table into a TimescaleDB hypertable
      -- Partitions by 'created_at' column (time-based)


      -- COMPRESSION (90% storage reduction)
  -- Old events are compressed automatically after 7 days
  -- Enable compression on the events table
  -- compress_segmentby: Keep same aggregate's events together when compressing
   ALTER TABLE events SET (
      timescaledb.compress,
      timescaledb.compress_segmentby = 'aggregate_type, aggregate_id'
  );
 -- Automatically compress data older than 7 days
  -- Recent data: fast writes (uncompressed)
  -- Old data: fast reads, 90% smaller (compressed)
    SELECT add_compression_policy('events', INTERVAL '7 days');
  -- OUTBOX PATTERN
  -- Guarantees events reach Kafka even if Kafka is temporarily down
  CREATE TABLE outbox (
      -- Auto-incrementing ID (order matters for publishing)
      id BIGSERIAL PRIMARY KEY,

      -- Same fields as events table for routing
      aggregate_type VARCHAR(50) NOT NULL,
      aggregate_id UUID NOT NULL,
      event_type VARCHAR(100) NOT NULL,

      -- The event payload to send to Kafka
      payload JSONB NOT NULL,

      -- When was this outbox entry created?
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

      -- When was it published to Kafka?
      -- NULL = not yet published (worker will pick it up)
      -- NOT NULL = already sent to Kafka
      published_at TIMESTAMPTZ
  );
 -- Index for finding unpublished events quickly
  -- "WHERE published_at IS NULL" is the hot query
  CREATE INDEX idx_outbox_unpublished
      ON outbox (created_at)
      WHERE published_at IS NULL;
       
       
-- READ MODELS (Denormalized for Dashboard Queries)
  -- Updated by Kafka consumers when events are processed
  -- Current account balances (projected from events)
  CREATE TABLE account_projections (
      account_id UUID PRIMARY KEY,
      user_id UUID NOT NULL,
      bank_name VARCHAR(100),
      account_type VARCHAR(50),        -- 'checking', 'savings', 'credit'
      currency VARCHAR(3) DEFAULT 'NOK',
      current_balance DECIMAL(15, 2),  -- DECIMAL for money, never FLOAT!
      last_synced_at TIMESTAMPTZ,
      last_event_version INTEGER,      -- Track which events we've processed
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  -- Daily spending aggregates (for charts)
  CREATE TABLE daily_aggregates (
      id BIGSERIAL,
      user_id UUID NOT NULL,
      date DATE NOT NULL,
      category VARCHAR(100) NOT NULL,
      total_amount DECIMAL(15, 2) NOT NULL,
      transaction_count INTEGER NOT NULL,
      avg_amount DECIMAL(15, 2),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

      -- Composite primary key includes date for hypertable
      PRIMARY KEY (id, date),

      -- One row per user/date/category combination
      UNIQUE (user_id, date, category)
  );

  -- Convert to hypertable (partitioned by date)
  SELECT create_hypertable('daily_aggregates', 'date');

  -- Budget tracking with computed columns
  CREATE TABLE budget_status (
      id BIGSERIAL PRIMARY KEY,
      user_id UUID NOT NULL,
      category VARCHAR(100) NOT NULL,
      month DATE NOT NULL,                    -- First day of month
      budget_amount DECIMAL(15, 2) NOT NULL,
      spent_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,

      -- GENERATED columns: PostgreSQL calculates automatically!
      -- No need to update these manually
      remaining_amount DECIMAL(15, 2)
          GENERATED ALWAYS AS (budget_amount - spent_amount) STORED,

      percentage_used DECIMAL(5, 2)
          GENERATED ALWAYS AS (
              CASE WHEN budget_amount > 0
                   THEN (spent_amount / budget_amount * 100)
                   ELSE 0
              END
          ) STORED,

      alert_threshold_reached BOOLEAN DEFAULT FALSE,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

      UNIQUE (user_id, category, month)
  );
 -- PostgreSQL automatically calculates:
  -- remaining_amount = 1000 - 850 = 150
  -- percentage_used = 850/1000 * 100 = 85.00
  -- No application code needed!

    -- GDPR COMPLIANCE
  -- Required for any European financial application
  -- User consent records (GDPR Article 7)
  CREATE TABLE user_consents (
      id BIGSERIAL PRIMARY KEY,
      user_id UUID NOT NULL,

      -- What did they consent to?
      consent_type VARCHAR(50) NOT NULL,  -- 'data_processing', 'marketing', 'analytics'

      -- Did they agree?
      granted BOOLEAN NOT NULL,
      granted_at TIMESTAMPTZ,
      revoked_at TIMESTAMPTZ,             -- NULL if still active

      -- Which version of privacy policy?
      policy_version VARCHAR(20) NOT NULL,

      -- Evidence of consent
      ip_address INET,                    -- PostgreSQL's IP address type
      user_agent TEXT
  );

  -- Audit log: who accessed what data? (GDPR Article 30)
  CREATE TABLE audit_log (
      id BIGSERIAL,
      user_id UUID,                       -- Who performed the action

      action VARCHAR(50) NOT NULL,        -- 'view', 'export', 'delete', 'modify'
      resource_type VARCHAR(50) NOT NULL, -- 'transaction', 'account', 'profile'
      resource_id UUID,

      details JSONB,                      -- Additional context
      ip_address INET,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );
    SELECT create_hypertable('audit_log', 'created_at');
 -- INDEXES (Speed up common queries)
  -- Find all events for an aggregate (rebuilding state)
  CREATE INDEX idx_events_aggregate
      ON events (aggregate_type, aggregate_id, version);

  -- Dashboard: "Show my transactions from last month"
  CREATE INDEX idx_daily_agg_user_date
      ON daily_aggregates (user_id, date DESC);

  -- Budget page: "Show my current month's budgets"
  CREATE INDEX idx_budget_user_month
      ON budget_status (user_id, month DESC);

  -- GDPR: "Find all consents for a user"
  CREATE INDEX idx_consents_user
      ON user_consents (user_id);

  -- Audit: "What did this user do?"
  CREATE INDEX idx_audit_user
      ON audit_log (user_id, created_at DESC);
    -- Index for finding events by type (analytics)
  CREATE INDEX idx_events_type
      ON events (event_type, created_at DESC);

  -- Auto-update updated_at timestamps
  CREATE OR REPLACE FUNCTION update_updated_at_column()
  RETURNS TRIGGER AS $$
  BEGIN
      NEW.updated_at = NOW();
      RETURN NEW;
  END;
  $$ language 'plpgsql';

  CREATE TRIGGER update_account_projections_updated_at
      BEFORE UPDATE ON account_projections
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();

  CREATE TRIGGER update_budget_status_updated_at
      BEFORE UPDATE ON budget_status
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();

     -- Organizations (multi-tenant)
  CREATE TABLE organizations (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      name VARCHAR(255) NOT NULL,
      slug VARCHAR(100) UNIQUE NOT NULL,  -- "techstartup-as"
      plan VARCHAR(50) DEFAULT 'free',     -- 'free', 'pro', 'enterprise'
      settings JSONB DEFAULT '{}',
      created_at TIMESTAMPTZ DEFAULT NOW()
  );

  -- Departments within organization
  CREATE TABLE departments (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      name VARCHAR(255) NOT NULL,
      budget_monthly DECIMAL(15, 2),
      parent_department_id UUID,  -- For hierarchy
      created_at TIMESTAMPTZ DEFAULT NOW()
  );

  -- Projects (cross-department)
  CREATE TABLE projects (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      name VARCHAR(255) NOT NULL,
      budget_total DECIMAL(15, 2),
      start_date DATE,
      end_date DATE,
      status VARCHAR(50) DEFAULT 'active'
  );

  -- Cost Centers (accounting codes)
  CREATE TABLE cost_centers (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      code VARCHAR(50) NOT NULL,  -- "CC-001"
      name VARCHAR(255) NOT NULL,
      budget_yearly DECIMAL(15, 2)
  );

  -- Organization members with roles
  CREATE TABLE organization_members (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      user_id UUID NOT NULL,
      role VARCHAR(50) NOT NULL,  -- 'owner', 'admin', 'finance', 'employee'
      department_id UUID REFERENCES departments(id),
      spending_limit DECIMAL(15, 2),  -- Personal spending limit
      can_approve_up_to DECIMAL(15, 2),  -- Approval authority
      created_at TIMESTAMPTZ DEFAULT NOW(),

      UNIQUE(organization_id, user_id)
  );

  -- Expenses (enterprise expense tracking)
  CREATE TABLE expenses (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      submitted_by UUID NOT NULL,
      amount DECIMAL(15, 2) NOT NULL,
      currency VARCHAR(3) DEFAULT 'NOK',
      category VARCHAR(100),
      description TEXT,
      merchant_name VARCHAR(255),
      expense_date DATE,

      -- Allocation
      department_id UUID REFERENCES departments(id),
      project_id UUID REFERENCES projects(id),
      cost_center_id UUID REFERENCES cost_centers(id),

      -- Approval workflow
      status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'approved', 'rejected', 'paid'
      approved_by UUID,
      approved_at TIMESTAMPTZ,
      rejection_reason TEXT,

      -- Receipt/Invoice
      receipt_url TEXT,
      ocr_data JSONB,  -- ML-extracted data from receipt

      created_at TIMESTAMPTZ DEFAULT NOW()
  );

  -- Approval workflow rules
  CREATE TABLE approval_rules (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      organization_id UUID REFERENCES organizations(id),
      name VARCHAR(255),
      condition_type VARCHAR(50),  -- 'amount_above', 'category', 'department'
      condition_value JSONB,
      approver_role VARCHAR(50),   -- Who can approve
      auto_approve BOOLEAN DEFAULT FALSE,
      priority INTEGER DEFAULT 0,

      UNIQUE(organization_id, priority)
  );
"""
Locust performance test for Finance Tracker API.

Each virtual user registers its own account on startup,
then hammers the hot read paths and occasionally writes.

Run (headless):
    locust -f tests/performance/locustfile.py --host=http://localhost:8000 --headless -u 50 -r 5 --run-time 60s --csv=tests/performance/results

Run (web UI at http://localhost:8089):
    locust -f tests/performance/locustfile.py --host=http://localhost:8000
"""
import random
import uuid
from locust import HttpUser, task, between


class FinanceTrackerUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        """Register + login once per virtual user. Each gets its own org."""
        uid = uuid.uuid4().hex[:8]
        self.email = f"perf-{uid}@loadtest.no"
        self.password = "PerfTest123!"

        # Register (creates user + org atomically)
        reg = self.client.post("/api/v1/auth/register", json={
            "email": self.email,
            "password": self.password,
            "full_name": f"Perf User {uid}",
            "org_name": f"Perf Org {uid}",
        }, name="/auth/register")

        if reg.status_code != 201:
            self.environment.runner.quit()
            return

        self.headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        self.account_id = None  # populated on first create_transaction

    @task(5)
    def list_transactions(self):
        self.client.get("/api/v1/transactions/", headers=self.headers, name="/transactions/ [GET]")

    @task(3)
    def get_account(self):
        if self.account_id:
            self.client.get(f"/api/v1/accounts/{self.account_id}", headers=self.headers, name="/accounts/{id} [GET]")

    @task(2)
    def list_integrations(self):
        self.client.get("/api/v1/integrations/", headers=self.headers, name="/integrations/ [GET]")

    @task(1)
    def create_transaction(self):
        # Lazily create an account on first write, reuse after
        if self.account_id is None:
            self._create_account()
        if self.account_id is None:
            return  # account creation failed, skip

        self.client.post("/api/v1/transactions/", headers=self.headers, json={
            "account_id": self.account_id,
            "amount": str(round(random.uniform(10, 50000), 2)),
            "currency": "NOK",
            "merchant_name": "Perf Test Merchant",
            "description": "Perf test transaction",
            "transaction_type": "debit",
        }, name="/transactions/ [POST]")

    def _create_account(self):
        """POST a new account and store its ID for subsequent reads/writes."""
        resp = self.client.post("/api/v1/accounts/", headers=self.headers, json={
            "name": "Perf Test Account",
            "account_type": "CHECKING",
            "currency": "NOK",
            "initial_balance": "10000.00",
        }, name="/accounts/ [POST]")
        if resp.status_code in (200, 201):
            self.account_id = str(resp.json().get("account_id"))

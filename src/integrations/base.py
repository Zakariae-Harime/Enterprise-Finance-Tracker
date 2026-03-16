from abc import ABC, abstractmethod
from typing import List

from src.integrations.models import ERPExpense, ERPInvoice, SyncResult
from src.infrastructure.circuit_breaker import CircuitBreaker, CircuitOpenError, BREAKERS


class ERPAdapter(ABC):
    """
    Abstract base class for ERP adapters.

    Each concrete adapter (Tripletex, SAP, Dynamics) implements _push_expense()
    and _pull_invoices() with the actual HTTP calls.

    The public push_expense() and pull_invoices() methods on this base class
    wrap those implementations with a circuit breaker, so subclasses get
    fail-fast protection automatically without any extra code.

    Pattern: Template Method
      Base class defines the algorithm skeleton (circuit breaker wrap).
      Subclasses fill in the steps (_push_expense, _pull_invoices).
    """

    # Each subclass sets this to the key in BREAKERS that matches the ERP.
    # Example: TripletexAdapter sets erp_name = "tripletex"
    # This tells the base class which circuit breaker to use.
    erp_name: str = "unknown"

    @property
    def _breaker(self) -> CircuitBreaker:
        """
        Return the circuit breaker for this ERP.

        Looks up BREAKERS by erp_name. If the subclass didn't set erp_name,
        or set it to an unknown name, returns a default permissive breaker
        with a very high threshold (effectively disabled) so the adapter
        still works even without a registered breaker.
        """
        return BREAKERS.get(self.erp_name) or CircuitBreaker(
            name=self.erp_name,
            failure_threshold=999,   # effectively disabled
            recovery_timeout=60.0,
        )

    # ── Public interface (circuit-breaker wrapped) ────────────────────────────

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        """
        Push an expense to the ERP, protected by a circuit breaker.

        If the circuit is OPEN (ERP is known to be down), returns a failed
        SyncResult immediately without making any network call.

        If the circuit is CLOSED or HALF-OPEN, delegates to _push_expense()
        (implemented by the concrete subclass). Any exception from _push_expense()
        is caught by the breaker (counts as a failure) and converted to a
        failed SyncResult so callers never need to handle exceptions here.
        """
        try:
            # breaker.call() handles the state machine.
            # It calls self._push_expense(expense) if state allows,
            # or raises CircuitOpenError immediately if OPEN.
            return await self._breaker.call(self._push_expense, expense)

        except CircuitOpenError as e:
            # ERP is known down — return a clear failure result.
            # The caller (expense sync service) can log this and move on
            # without hanging for a timeout.
            return SyncResult(
                success=False,
                error=f"Circuit OPEN — {self.erp_name} temporarily unavailable: {e}",
            )

        except Exception as e:
            # The actual HTTP call failed. Breaker already counted this failure.
            # Return a descriptive SyncResult so the caller knows what went wrong.
            return SyncResult(
                success=False,
                error=f"{self.erp_name} push failed: {e}",
            )

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        """
        Pull invoices from the ERP, protected by a circuit breaker.

        Returns an empty list if the circuit is OPEN or the call fails,
        so callers can always iterate the result without checking for None.
        """
        try:
            return await self._breaker.call(self._pull_invoices, since_date)

        except CircuitOpenError:
            return []   # ERP down — no invoices to pull, try again later

        except Exception:
            return []   # network error — return empty, breaker counted the failure

    # ── Abstract methods (implemented by each concrete ERP adapter) ───────────

    @abstractmethod
    async def _push_expense(self, expense: ERPExpense) -> SyncResult:
        """
        Make the actual HTTP call to push an expense to the ERP.

        Implemented by TripletexAdapter, SAPAdapter, DynamicsAdapter.
        MUST NOT catch exceptions — let them propagate so the circuit
        breaker can count them as failures.
        Never call this directly — always go through push_expense().
        """
        ...

    @abstractmethod
    async def _pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        """
        Make the actual HTTP call to pull invoices from the ERP.

        Same rules as _push_expense — don't catch exceptions here.
        """
        ...

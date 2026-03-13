from src.accounting.tripletex import TripletexAdapter
from src.accounting.visma import VismaAdapter
from src.accounting.xero import XeroAdapter
from src.accounting.quickbooks import QuickBooksAdapter
from src.erp.sap import SAPAdapter
from src.erp.dynamics import DynamicsAdapter
from src.integrations.base import ERPAdapter

ADAPTERS: dict[str, type[ERPAdapter]] = {
    "tripletex":  TripletexAdapter,
    "visma":      VismaAdapter,
    "xero":       XeroAdapter,
    "quickbooks": QuickBooksAdapter,
    "sap":        SAPAdapter,
    "dynamics":   DynamicsAdapter,
}


def get_adapter(provider: str) -> type[ERPAdapter]:
    if provider not in ADAPTERS:
        raise ValueError(f"Unknown ERP provider: '{provider}'. Supported: {list(ADAPTERS)}")
    return ADAPTERS[provider]

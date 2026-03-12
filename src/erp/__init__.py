from src.accounting.tripletex import TripletexAdapter
from src.erp.sap import SAPAdapter
from src.erp.dynamics import DynamicsAdapter
from src.integrations.base import ERPAdapter

ADAPTERS: dict[str, type[ERPAdapter]] = {
    "tripletex": TripletexAdapter,
    "sap":       SAPAdapter,
    "dynamics":  DynamicsAdapter,
}


def get_adapter(provider: str) -> type[ERPAdapter]:
    if provider not in ADAPTERS:
        raise ValueError(f"Unknown ERP provider: '{provider}'. Supported: {list(ADAPTERS)}")
    return ADAPTERS[provider]

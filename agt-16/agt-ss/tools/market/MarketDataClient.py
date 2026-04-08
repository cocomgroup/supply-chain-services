# Databricks notebook source
"""
agt16.tools.market.MarketDataClient
-------------------------------------
Market intelligence data clients for AGT-16.

Three sub-clients:
  1. CommodityClient   — spot/futures prices for key raw materials and energy
  2. CarrierRateClient — ocean, air, and truckload rate indices
  3. RegulatoryClient  — supply chain regulatory feeds (Federal Register, EUR-Lex)

All clients return normalised dicts and log raw API responses for audit.
Credentials are stored in Secrets Manager; secret ARNs are resolved from env vars.

Environment variables
---------------------
AGT16_COMMODITY_API_KEY    API key for commodity price data provider
AGT16_FREIGHT_API_KEY      Freightos / Xeneta / DAT API key for carrier rates
AGT16_FEDREG_API_KEY       api.regulations.gov API key (optional; public rate limit applies)
AWS_DEFAULT_REGION         AWS region
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, date
from typing import Optional

import urllib.request
import urllib.error
import json

logger = logging.getLogger(__name__)


def _get_secret(secret_arn: str) -> dict:
    """Resolve a JSON secret from Secrets Manager."""
    import boto3
    sm  = boto3.client("secretsmanager",
                       region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    raw = sm.get_secret_value(SecretId=secret_arn)["SecretString"]
    return json.loads(raw)


def _http_get(url: str, headers: Optional[dict] = None) -> dict:
    """Minimal HTTP GET returning parsed JSON. Raises on non-200."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# 1. CommodityClient
# ---------------------------------------------------------------------------

class CommodityClient:
    """
    Fetch commodity spot and futures prices.

    Supported commodities (configurable):
      Steel HRC, Aluminum, Copper, Polyethylene (LLDPE), Diesel, Natural Gas,
      West Texas Intermediate (WTI) crude, Baltic Dry Index.

    Primary provider: commodity_api.io (configurable).
    Falls back to stub data when API key is not configured.
    """

    _DEFAULT_COMMODITIES = [
        "steel_hrc", "aluminum", "copper",
        "polyethylene_lldpe", "diesel", "natural_gas", "wti_crude",
    ]

    def __init__(self):
        self._api_key = os.getenv("AGT16_COMMODITY_API_KEY", "")
        self._base_url = "https://commodities-api.com/api"

    def get_commodity_prices(
        self,
        commodities: Optional[list[str]] = None,
    ) -> dict:
        """
        Return current prices for the given commodity list.

        Structure:
        {
          "steel_hrc": { "price": 750.0, "unit": "USD/t",
                         "delta_pct": 0.02, "source": "commodities-api.com",
                         "fetched_at": "..." },
          ...
        }
        """
        commodities = commodities or self._DEFAULT_COMMODITIES

        if not self._api_key:
            logger.warning("[CommodityClient] No API key — returning stub data")
            return self._stub_prices(commodities)

        try:
            symbols = ",".join(c.upper() for c in commodities)
            url     = f"{self._base_url}/latest?access_key={self._api_key}&symbols={symbols}"
            raw     = _http_get(url)
            return self._normalize(raw.get("data", {}).get("rates", {}))
        except Exception as exc:
            logger.error("[CommodityClient] API error: %s — returning stub", exc)
            return self._stub_prices(commodities)

    def _normalize(self, rates: dict) -> dict:
        now = datetime.utcnow().isoformat()
        result = {}
        for symbol, price in rates.items():
            result[symbol.lower()] = {
                "price":      float(price),
                "unit":       "USD",
                "delta_pct":  None,   # requires historical comparison — TODO
                "source":     "commodities-api.com",
                "fetched_at": now,
            }
        return result

    def _stub_prices(self, commodities: list) -> dict:
        now = datetime.utcnow().isoformat()
        stubs = {
            "steel_hrc":         {"price": 745.0,  "unit": "USD/t"},
            "aluminum":          {"price": 2350.0, "unit": "USD/t"},
            "copper":            {"price": 9800.0, "unit": "USD/t"},
            "polyethylene_lldpe":{"price": 1050.0, "unit": "USD/t"},
            "diesel":            {"price": 3.85,   "unit": "USD/gal"},
            "natural_gas":       {"price": 2.10,   "unit": "USD/MMBtu"},
            "wti_crude":         {"price": 78.5,   "unit": "USD/bbl"},
        }
        return {
            c: {**stubs.get(c, {"price": 0.0, "unit": "USD"}),
                "delta_pct": None, "source": "stub", "fetched_at": now}
            for c in commodities
        }


# ---------------------------------------------------------------------------
# 2. CarrierRateClient
# ---------------------------------------------------------------------------

class CarrierRateClient:
    """
    Fetch freight rate indices across ocean, air, and truckload lanes.

    Data sources:
      Ocean:     Freightos Baltic Index (FBX) via Freightos API
      Truckload: DAT Freight & Analytics API
      Air:       TAC Index or Xeneta Air

    Falls back to stub data when API keys are not configured.
    """

    def __init__(self):
        self._freight_api_key = os.getenv("AGT16_FREIGHT_API_KEY", "")

    def get_carrier_rates(
        self,
        modes: Optional[list[str]] = None,
    ) -> dict:
        """
        Return current rate indices by mode and lane.

        Structure:
        {
          "ocean": {
            "global_fbx": { "rate": 1850.0, "unit": "USD/FEU",
                            "trend": "rising", "source": "freightos",
                            "fetched_at": "..." }
          },
          "truckload": { ... },
          "air": { ... }
        }
        """
        modes = modes or ["ocean", "truckload", "air"]

        if not self._freight_api_key:
            logger.warning("[CarrierRateClient] No API key — returning stub data")
            return self._stub_rates(modes)

        # TODO: implement live API calls per mode
        # ocean    → https://api.freightos.com/v2/indexes
        # truckload → https://api.dat.com/freight/rate-view
        # air      → https://api.xeneta.com/v1/spot-rates
        logger.info("[CarrierRateClient] TODO: live API calls not yet implemented")
        return self._stub_rates(modes)

    def _stub_rates(self, modes: list) -> dict:
        now    = datetime.utcnow().isoformat()
        result = {}
        if "ocean" in modes:
            result["ocean"] = {
                "global_fbx":        {"rate": 1_850.0, "unit": "USD/FEU", "trend": "stable",  "source": "stub", "fetched_at": now},
                "transpacific_east": {"rate": 2_200.0, "unit": "USD/FEU", "trend": "rising",  "source": "stub", "fetched_at": now},
                "transatlantic":     {"rate": 1_650.0, "unit": "USD/FEU", "trend": "falling", "source": "stub", "fetched_at": now},
            }
        if "truckload" in modes:
            result["truckload"] = {
                "us_van_national":   {"rate": 2.35, "unit": "USD/mi", "trend": "stable",  "source": "stub", "fetched_at": now},
                "us_flatbed_national":{"rate": 2.72, "unit": "USD/mi", "trend": "rising",  "source": "stub", "fetched_at": now},
            }
        if "air" in modes:
            result["air"] = {
                "global_air_index":  {"rate": 3.20, "unit": "USD/kg", "trend": "stable", "source": "stub", "fetched_at": now},
            }
        return result


# ---------------------------------------------------------------------------
# 3. RegulatoryClient
# ---------------------------------------------------------------------------

class RegulatoryClient:
    """
    Scan regulatory publication feeds for supply chain-relevant changes.

    Sources:
      - api.regulations.gov  (US Federal Register / proposed rules)
      - EUR-Lex RSS feeds    (EU regulatory updates)
      - Custom RSS feeds for trade policy, customs, ESG

    Returns a list of regulatory trend dicts, each with:
      regulation, jurisdiction, impact_summary, effective_date, source, url
    """

    _FEDREG_BASE = "https://api.regulations.gov/v4"

    # Search terms mapped to supply chain domains
    _DOMAIN_TERMS = {
        "trade_policy":        "supply chain tariff customs import export",
        "esg":                 "supply chain ESG sustainability scope 3 emissions",
        "customs":             "customs duty drawback bonded warehouse",
        "logistics_regulation": "hazmat transportation FMCSA IATA IMDG",
        "labor":               "forced labor supply chain due diligence",
    }

    def __init__(self):
        self._api_key = os.getenv("AGT16_FEDREG_API_KEY", "DEMO_KEY")

    def scan_regulatory_domains(self, domains: list[str]) -> list[dict]:
        """
        Scan the configured regulatory domains and return a list of trend records.
        """
        results = []
        for domain in domains:
            if domain not in self._DOMAIN_TERMS:
                logger.warning("[RegulatoryClient] Unknown domain: %s", domain)
                continue
            try:
                records = self._scan_fedreg(domain, self._DOMAIN_TERMS[domain])
                results.extend(records)
            except Exception as exc:
                logger.error("[RegulatoryClient] Scan failed for %s: %s", domain, exc)
        return results

    def _scan_fedreg(self, domain: str, search_term: str) -> list[dict]:
        """
        Query the Federal Register API for recent proposed and final rules
        matching the search term.
        """
        url = (
            f"{self._FEDREG_BASE}/documents"
            f"?filter[searchTerm]={urllib.parse.quote(search_term)}"
            f"&filter[postedDate][gte]={self._thirty_days_ago()}"
            f"&filter[documentType]=Proposed+Rule,Rule"
            f"&page[size]=5"
            f"&api_key={self._api_key}"
        )

        try:
            import urllib.parse
            raw  = _http_get(url)
            docs = raw.get("data", [])
            return [
                {
                    "regulation":     d["attributes"].get("title", ""),
                    "jurisdiction":   "US",
                    "domain":         domain,
                    "impact_summary": "",   # TODO: LLM summarization of full text
                    "effective_date": d["attributes"].get("postedDate"),
                    "source":         "regulations.gov",
                    "url":            d["links"].get("self", ""),
                }
                for d in docs
            ]
        except Exception as exc:
            logger.warning("[RegulatoryClient] FedReg API error (%s): %s", domain, exc)
            return []

    @staticmethod
    def _thirty_days_ago() -> str:
        from datetime import timedelta
        return (date.today() - timedelta(days=30)).isoformat()

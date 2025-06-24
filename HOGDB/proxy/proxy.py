# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

import requests
from typing import Dict, Optional, List


class ProxyDriver:
    def __init__(self, db_uri: str, db_username: str, db_password: str, proxy_url: str):
        self.url = proxy_url
        response = requests.post(
            f"{self.url}/initialize",
            json={
                "db_uri": db_uri,
                "db_username": db_username,
                "db_password": db_password,
            },
        )
        response.raise_for_status()
        self.driver_id = response.json().get("id")

    def verify_connectivity(self):
        response = requests.get(f"{self.url}/ping", params={"id": self.driver_id})
        response.raise_for_status()

    def close(self):
        response = requests.get(f"{self.url}/close", params={"id": self.driver_id})
        response.raise_for_status()

    def session(self, database: str = "neo4j"):
        return ProxySession(self.driver_id, self.url, database)


class ProxySession:
    def __init__(self, driver_id: int, proxy_url: str, db_name: str = "neo4j"):
        self.proxy_url = proxy_url
        self.db_name = db_name
        self.driver_id = driver_id
        response = requests.post(
            f"{self.proxy_url}/session/init",
            json={"database": self.db_name, "id": self.driver_id},
        )
        response.raise_for_status()

    def run(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        payload = {
            "id": self.driver_id,
            "query": query,
            "parameters": parameters or {},
        }
        response = requests.post(f"{self.proxy_url}/session/run", json=payload)
        response.raise_for_status()
        return response.json().get("results", [])

    def begin_transaction(self):
        return ProxyTransaction(self.driver_id, self.proxy_url)

    def close(self):
        response = requests.get(
            f"{self.proxy_url}/session/close", params={"id": self.driver_id}
        )
        response.raise_for_status()


class ProxyTransaction:
    def __init__(self, driver_id: int, proxy_url: str):
        self.proxy_url = proxy_url
        response = requests.get(
            f"{self.proxy_url}/transaction/init", params={"id": driver_id}
        )
        self.driver_id = driver_id
        response.raise_for_status()

    def run(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        payload = {"id": self.driver_id, "query": query, "parameters": parameters or {}}
        response = requests.post(f"{self.proxy_url}/transaction/run", json=payload)
        response.raise_for_status()
        return response.json().get("results", [])

    def commit(self):
        response = requests.get(
            f"{self.proxy_url}/transaction/commit", params={"id": self.driver_id}
        )
        response.raise_for_status()

    def close(self):
        response = requests.get(
            f"{self.proxy_url}/transaction/close", params={"id": self.driver_id}
        )
        response.raise_for_status()

# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from flask import Flask, request, jsonify
from neo4j import GraphDatabase


app = Flask(__name__)

drivers = []
sessions = []
txs = []
active = 0


@app.route("/initialize", methods=["POST"])
def initialize_driver():
    global drivers, sessions, txs, active
    data = request.get_json()
    db_uri = data.get("db_uri")
    db_username = data.get("db_username")
    db_password = data.get("db_password")
    if not db_uri or not db_username or not db_password:
        return jsonify({"error": "Missing required parameters"}), 400
    drivers.append(GraphDatabase.driver(db_uri, auth=(db_username, db_password)))
    sessions.append(None)
    txs.append(None)
    active += 1
    return jsonify({"status": "initialized", "id": len(drivers) - 1}), 200


@app.route("/ping", methods=["GET"])
def ping():
    global drivers
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    try:
        drivers[driver_id].verify_connectivity()
        return jsonify({"status": "connected"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/close", methods=["GET"])
def close_driver():
    global drivers, sessions, txs, active
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    drivers[driver_id].close()
    drivers[driver_id] = None
    active -= 1
    if active == 0:
        drivers, sessions, txs = [], [], []
    return jsonify({"status": "closed"}), 200


@app.route("/session/init", methods=["POST"])
def initialize_session():
    global session, driver
    data = request.get_json()
    driver_id = data.get("id")
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    name = data.get("database", "neo4j")
    sessions[driver_id] = drivers[driver_id].session(database=name)
    return jsonify({"status": "session initialized", "database": name}), 200


@app.route("/session/run", methods=["POST"])
def run_session_query():
    global session
    data = request.get_json()
    driver_id = data.get("id")
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    query = data.get("query")
    parameters = data.get("parameters", {})
    results = [record for record in sessions[driver_id].run(query, parameters)]
    return jsonify({"results": results}), 200


@app.route("/session/close", methods=["GET"])
def close_session():
    global session
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    sessions[driver_id].close()
    sessions[driver_id] = None
    return jsonify({"status": "session closed"}), 200


@app.route("/transaction/init", methods=["GET"])
def initialize_transaction():
    global sessions, txs
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    txs[driver_id] = sessions[driver_id].begin_transaction()
    return jsonify({"status": "transaction initialized"}), 200


@app.route("/transaction/run", methods=["POST"])
def run_transaction_query():
    global txs
    data = request.get_json()
    driver_id = data.get("id")
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    query = data.get("query")
    parameters = data.get("parameters", {})
    results = [record for record in txs[driver_id].run(query, parameters)]
    return jsonify({"results": results}), 200


@app.route("/transaction/commit", methods=["GET"])
def commit_transaction():
    global txs
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    txs[driver_id].commit()
    return jsonify({"status": "transaction committed"}), 200


@app.route("/transaction/close", methods=["GET"])
def close_transaction():
    global txs
    driver_id = request.args.get("id", type=int)
    if driver_id is None or driver_id >= len(drivers):
        return jsonify({"error": "Invalid driver ID"}), 400
    txs[driver_id].close()
    txs[driver_id] = None
    return jsonify({"status": "transaction closed"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


import os

from langchain.agents import  tool, Tool
from langchain.pydantic_v1 import BaseModel, Field
import requests
from typing import Optional
import json
from typing import Any, Dict



data = {'dummy': 'dummy'} # A mock of the data

class CancelPendingOrderInput(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    reason: str = Field(description="The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.")

@tool("cancel_pending_order", args_schema=CancelPendingOrderInput)
def cancel_pending_order(order_id: str, reason: str)-> str:
    """Cancel a pending order. If the order is already processed or delivered,
    it cannot be cancelled. The agent needs to explain the cancellation detail
    the order status will be changed to 'cancelled' and the payment will be refunded.
    The refund will be added to the user's gift card balance immediately if the payment
    was made using a gift card, otherwise the refund would take 5-7 business days to process.
    The function returns the order details after the cancellation."""

    orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            payment_method = data["users"][order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    return json.dumps(order)


@tool
def parse_yaml_code(yaml_code: str) -> dict:
    """You must use this tool before sending the final output, the input is the yaml code with the output schema. The result is the final output!"""
    return "The Yaml doesn't have a valid yaml structure, please fix it such that it can be parsed. Remember that if you have a value that is a string, you should wrap it in quotes."
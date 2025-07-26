from . import growwapi
from langchain_core.tools import tool

@tool
def get_holdings_for_user(groww_symbol: str):
    """Get portfolio details ).
    The response contains the following fields:

Name	Type	Description
isin	string	The ISIN (International Securities Identification number) of the symbol
trading_symbol	string	The trading symbol of the holding
quantity	float	The net quantity of the holding
average_price	int	The average price of the holding
pledge_quantity	float	The pledged quantity of the holding
demat_locked_quantity	float	The demat locked quantity of the holding
groww_locked_quantity	float	The Groww locked quantity of the holding
repledge_quantity	float	The repledged quantity of the holding
t1_quantity	float	The T1 quantity of the holding
demat_free_quantity	float	The demat free quantity of the holding
corporate_action_additional_quantity	int	The corporate action additional quantity of the holding
active_demat_transfer_quantity	int	The active demat transfer quantity of the holding

    """
    print("get_holdings_for_user called!!")
    resp = growwapi.get_holdings_for_user(timeout=5)
    print(resp)
    return resp
# region imports
from AlgorithmImports import *
# endregion

#next steps: 
#trade only options with large implied volatility
#optimize butterfly wing spread 
#find good exit time

class ODTE_Options_Research(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_start_date(2024, 1, 10)
        self.set_cash(100000)
        #The following algorithm selects 0DTE Option contracts for the SPY that fall within 3 strikes of the underlying price.
        option = self.add_option('SPY', Resolution.MINUTE)
        self._symbol = option.Symbol
        option.set_filter(self._filter)

    #iv: 0-20%

    def _filter(self, universe):
        #iron_butterfly(min_days_till_expiry: int, strike_spread: float)
        return universe.include_weeklys().iv(0, 20).expiration(0, 0).iron_butterfly(0, 30)
        # #iron_condor(min_days_till_expiry: int, near_strike_spread: float, far_strike_spread: float)
        # return universe.include_weeklys().iv(0, 20).expiration(0, 0).iron_condor(30, 5, 10)



    def on_data(self, slice: Slice) -> None:
        if self.portfolio.invested:
            return

        # Get the OptionChain
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return

        self.debug(f'Current date: {self.Time.date()}')
        for x in chain:
            self.debug(f"Option symbol: {x.Symbol}, Type: {x.Right}, Strike: {x.Strike}, Expiry: {x.Expiry}")

        # Select expiry
        expiry = max([x.expiry for x in chain])
        self.debug(f"Expiry: {expiry}")

        # Separate the call and put contracts
        calls = [i for i in chain if i.right == OptionRight.CALL and i.expiry == expiry]
        puts = [i for i in chain if i.right == OptionRight.PUT and i.expiry == expiry]
        if not calls or not puts:
            return


        # Get the ATM and OTM strike prices
        atm_strike = sorted(calls, key = lambda x: abs(chain.underlying.price - x.strike))[0].strike
        otm_put_strike = min([x.strike for x in puts])
        otm_call_strike = 2 * atm_strike - otm_put_strike
        self.debug(f"ATM Strike: {atm_strike}, OTM Put Strike: {otm_put_strike}, OTM Call Strike: {otm_call_strike}")

        

        iron_butterfly = OptionStrategies.iron_butterfly(self._symbol, otm_put_strike, atm_strike, otm_call_strike, expiry)
        self.debug(iron_butterfly)
        self.buy(iron_butterfly, 2) 
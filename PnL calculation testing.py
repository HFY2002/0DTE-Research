# region imports
from AlgorithmImports import *
# endregion
import numpy as np

class SpreadFillModel(ImmediateFillModel):
    def MarketFill(self, asset, order):
        fill = super().MarketFill(asset, order)
        if order.Direction == OrderDirection.Buy:
            fill.FillPrice = asset.AskPrice
        elif order.Direction == OrderDirection.Sell:
            fill.FillPrice = asset.BidPrice
        return fill

class Testing(QCAlgorithm):


    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2023, 1, 5)
        self.set_cash(10000)
        self.stop_loss_ratio = 1
        option = self.add_option("SPY", Resolution.HOUR)
        self._symbol = option.symbol

        self.original_val = 0
        self.final_val = 0
        self.position = None
        option.set_filter(lambda x: x.include_weeklys().Expiration(0, 0))
        
        #option.SetFillModel(SpreadFillModel())
        self.diffs = []
        self.trade = True
            # Set Brokerage Model to include transaction costs
        #self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
    
    class Position:
        def __init__(self, iron_butterfly, chain, net_premium_received_per_underlying_asset, max_profit, max_loss, m, fee,
                     K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM,
                     atm_strike, entry_time, stop_loss_ratio):
            self.iron_butterfly = iron_butterfly
            self.chain = chain
            self.net_premium_received_per_underlying_asset = net_premium_received_per_underlying_asset
            self.max_profit = max_profit
            self.max_loss = max_loss # is positive, needs to mult*-1 be be actual payoff val
            self.m = m
            self.fee = fee
            self.K_C_ATM = K_C_ATM
            self.K_P_ATM = K_P_ATM
            self.K_C_OTM = K_C_OTM
            self.K_P_OTM = K_P_OTM
            self.atm_strike = atm_strike
            self.entry_time = entry_time
            self.stop_loss_ratio = stop_loss_ratio
            self.stop_loss = self.stop_loss_ratio * self.max_loss # is positive, needs to mult*-1 be be actual payoff val
            # Calculate price bounds for stop loss
            self.lower_bound, self.upper_bound = self.calculate_price_bounds(-self.stop_loss)
            self.lower_break_even, self.upper_break_even = self.calculate_price_bounds(0)

        def payoff(self, iron_butterfly_prem_per_underlying_asset):
            # Original payoff calculation
            P_T = (-1*(iron_butterfly_prem_per_underlying_asset) +
                    (self.net_premium_received_per_underlying_asset) * self.m) - self.fee
            # Adjust payoff for stop loss
            P_T = np.maximum(P_T, -self.stop_loss) # Negative because loss is negative payoff

            return P_T

        def calculate_price_bounds(self, payoff_value):
            """
            Calculate the lower and upper bounds where the payoff equals the given value (e.g., stop-loss or break-even).
            """
            x1 = self.K_C_ATM
            x2 = self.K_P_OTM
            y1 = self.net_premium_received_per_underlying_asset*self.m - self.fee
            y2 = -self.max_loss
            y3 = payoff_value
            if x2 == x1:
                raise ValueError("x1 and x2 cannot be the same; vertical line slope is undefined.")
            m = (y2 - y1) / (x2 - x1)
            # Calculate x3 based on y3
            x3 = (y3 - y1) / m + x1
            return x3, 2*x1-x3

    def iron_butterfly_prem_per_underlying_asset(self, iron_butterfly, chain, enter):
        C_ATM = P_ATM = C_OTM = P_OTM = None
        # Identify ATM and OTM strikes from the legs
        for leg in iron_butterfly.OptionLegs:
            # Find the contract in the chain
            contract = next(
                (x for x in chain if x.Strike == leg.Strike and x.Right == leg.Right and x.Expiry.date() == leg.Expiration.date()),
                None
            )
            if contract is None:
                self.Debug(f"Contract not found for leg: Strike {leg.Strike}, Right {leg.Right}, Expiration {leg.Expiration}")
                continue
            if leg.Quantity == -1:  # Short position: Use bid price (money received)
                if enter:
                    option_price = contract.BidPrice
                else:
                    option_price = contract.AskPrice
            elif leg.Quantity == 1:  # Long position: Use ask price (money paid)
                if enter:
                    option_price = contract.AskPrice
                else:
                    option_price = contract.BidPrice

            if option_price < 0:
                self.Debug(f"Option price is negative for {contract.Symbol}, and it is {option_price}")

            # Map premiums and strikes based on leg type
            if leg.Quantity == -1 and leg.Right == OptionRight.Call:
                C_ATM = option_price
            elif leg.Quantity == -1 and leg.Right == OptionRight.Put:
                P_ATM = option_price
            elif leg.Quantity == 1 and leg.Right == OptionRight.Call:
                C_OTM = option_price
            elif leg.Quantity == 1 and leg.Right == OptionRight.Put:
                P_OTM = option_price
        # Check if all necessary premiums and strikes are collected

        if None in [C_ATM, P_ATM, C_OTM, P_OTM]:
            self.Debug("Incomplete data to compute metrics.")
            return None

        return (C_ATM + P_ATM) - (C_OTM + P_OTM)


    def compute_iron_butterfly_metrics(self, iron_butterfly, chain, atm_strike):
        m = 100  # Contract multiplier for options
        fee = 4  # Set fees according to your brokerage model
        # Initialize variables to store option premiums and strikes
        K_C_ATM = K_P_ATM = K_C_OTM = K_P_OTM = None
        # Identify ATM and OTM strikes from the legs
        for leg in iron_butterfly.OptionLegs:
            contract = next(
                (x for x in chain if x.Strike == leg.Strike and x.Right == leg.Right and x.Expiry.date() == leg.Expiration.date()),
                None
            )
            if contract is None:
                self.Debug(f"Contract not found for leg: Strike {leg.Strike}, Right {leg.Right}, Expiration {leg.Expiration}")
                continue
            # Map premiums and strikes based on leg type
            if leg.Quantity == -1 and leg.Right == OptionRight.Call: # Short position: Use bid price (money received)
                K_C_ATM = leg.Strike
            elif leg.Quantity == -1 and leg.Right == OptionRight.Put: # Short position: Use bid price (money received)
                K_P_ATM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Call: # Long position: Use ask price (money paid)
                K_C_OTM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Put: # Long position: Use ask price (money paid)
                K_P_OTM = leg.Strike

        # # Check if all necessary premiums and strikes are collected
        if None in [K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM]:
            self.Debug("Incomplete data to compute metrics.")
            return None

        # Calculate net premium received (credit received at position opening)
        net_premium_received_per_underlying_asset = self.iron_butterfly_prem_per_underlying_asset(iron_butterfly, chain, enter = True)
        max_profit = net_premium_received_per_underlying_asset*m - fee
        # Wing width (difference between OTM and ATM strikes)
        wing_width = K_C_OTM - K_C_ATM
        # Maximum loss (difference between wing width and net premium received, plus fees)
        max_loss = ((wing_width - net_premium_received_per_underlying_asset) * m) + fee  # Positive; multiply by -1 for payoff
        # Create Position object to store metrics
        position = self.Position(
            iron_butterfly=iron_butterfly,
            chain=chain,
            net_premium_received_per_underlying_asset=net_premium_received_per_underlying_asset,
            max_profit=max_profit,
            max_loss=max_loss,
            m=m,
            fee=fee,
            K_C_ATM=K_C_ATM,
            K_P_ATM=K_P_ATM,
            K_C_OTM=K_C_OTM,
            K_P_OTM=K_P_OTM,
            atm_strike=atm_strike,
            entry_time=self.Time,
            stop_loss_ratio=self.stop_loss_ratio
        )
        return position

    def on_data(self, slice):
        if self.trade == False:
            return

        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return
        # Select expiry
        expiry = max([x.expiry for x in chain])
        # Separate the call and put contracts
        calls = [i for i in chain if i.right == OptionRight.CALL and i.expiry == expiry]
        puts = [i for i in chain if i.right == OptionRight.PUT and i.expiry == expiry]
        if not calls or not puts:
            return
        available_strikes = sorted(set([contract.Strike for contract in chain]))

        # Get the ATM and OTM strike prices
        atm_strike = sorted(calls, key=lambda x: abs(x.strike - chain.underlying.price))[0].strike
        wing_spread = np.random.randint(1, 50)  # Note: the upper bound is exclusive
        if ((atm_strike - wing_spread) not in available_strikes) or ((atm_strike + wing_spread) not in available_strikes):
            return

        iron_butterfly = OptionStrategies.iron_butterfly(self._symbol, atm_strike - wing_spread, atm_strike, atm_strike + wing_spread, expiry)
        position = self.compute_iron_butterfly_metrics(iron_butterfly, chain, atm_strike)
        iron_butterfly_prem_per_underlying_asset = self.iron_butterfly_prem_per_underlying_asset(position.iron_butterfly, chain, enter = False)


        self.Debug(f"Max profit: {(position.max_profit)}, Payoff: {(position.payoff(iron_butterfly_prem_per_underlying_asset))}")
        self.Debug(f"Portfolio val, cash prior to entering: {(self.Portfolio.TotalPortfolioValue)}, {(self.Portfolio.Cash)}")
        self.buy(position.iron_butterfly, 1)
        self.Debug(f"Total Portfolio val, cash right after entering: {(self.Portfolio.TotalPortfolioValue)}, {(self.Portfolio.Cash)}")
        self.position = position


        iron_butterfly_prem_per_underlying_asset = self.iron_butterfly_prem_per_underlying_asset(position.iron_butterfly, chain, enter = False)
        net_cash_change = iron_butterfly_prem_per_underlying_asset*(-100) - 4

        self.Debug(f"Expected cash change, payoff from exiting the position {net_cash_change}, {(self.position.payoff(iron_butterfly_prem_per_underlying_asset))}")  # final goal is {profit+self.initial_portfolio_val}")
        self.Debug(f"Portfolio val, cash prior to exiting: {(self.Portfolio.TotalPortfolioValue)}, {(self.Portfolio.Cash)}")
        self.Liquidate()
        self.Debug(f"Portfolio val, cash after exiting: {(self.Portfolio.TotalPortfolioValue)}, {(self.Portfolio.Cash)}")
        self.trade = False

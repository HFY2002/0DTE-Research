from AlgorithmImports import *
from scipy import stats
from collections import deque
import numpy as np
from scipy.stats import t

class ODTE_Options_Research(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 5, 1)
        self.SetCash(1000000)

        # Add SPY option data with hourly resolution
        option = self.AddOption('SPY', Resolution.Hour)
        self._symbol = option.Symbol
        option.SetFilter(self._filter)

        # Running window for price data
        self.lookback_period = 65  # Lookback period for historical volatility
        self.price_window = deque(maxlen=self.lookback_period)

        # Running window for IV/HV ratios
        self.window_len = 200
        self.running_window = deque(maxlen=self.window_len)

        # Warm-up period for sufficient data collection
        self.SetWarmUp(self.window_len, Resolution.Hour)

        # Add SPY equity data
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol

        # Parameters
        self.kelly_norm_factor = 1
        self.stop_loss_ratio = 0.5  # Stop loss at 50%
        self.capital_before_investment = None
        self.no_trades = None
        self.param_dict = {
            10: (2.6919676639088532, 0.0005309834758547001, 0.005187973392625497),
            11: (2.45935445274916, 0.00047545495802637583, 0.004290066437038121),
            12: (1.8730134231612623, 0.00030047146446993426, 0.0034277997022314245),
            13: (1.8760251263220549, 0.00029422672233981716, 0.0030622330062556958),
            14: (1.89343793022821, 0.00024803667549057817, 0.002602885517628698),
            15: (1.9734008826364353, 0.000035561218106801466, 0.0019824631710342266),
        }

        self.position = None  # Initialize with no active position
        self.profit_from_exit = 0
        self.profit_from_expiry = 0
        self.portfolio.set_margin_call_model(DefaultMarginCallModel(self.portfolio, self.default_order_properties))
        self.invested = False

    def _filter(self, universe):
        """
        Filters option contracts to include only weekly expirations with 0 days to expiry.
        """
        return universe.IncludeWeeklys().Expiration(0, 0)

    def t_distribution_spread(self, degrees_of_freedom, mean, scale, percent):
        """
        Calculates the spread based on a t-distribution for a given confidence level.
        """
        alpha = (1 - percent / 100) / 2
        spread = (stats.t.ppf(1 - alpha, degrees_of_freedom) - stats.t.ppf(alpha, degrees_of_freedom)) * scale
        return spread

    def update_price_window(self, close_price):
        """
        Updates the rolling price window with the latest closing price.
        """
        self.price_window.append(close_price)

    def calculate_historical_volatility(self):
        """
        Calculates annualized historical volatility from the rolling price window.
        Returns None if there is insufficient data.
        """
        if len(self.price_window) < self.lookback_period:
            return None
        prices = np.array(list(self.price_window))
        returns = np.diff(prices) / prices[:-1]
        hv = np.std(returns) * (252 ** 0.5)  # Annualized volatility
        return hv

    def select_best_iron_butterfly(self, chain, atm_strike):
        """
        Constructs multiple iron butterfly positions with varying wing spreads,
        calculates the expected value for each, and returns the one with the highest EV.
        """
        expiry = max([x.Expiry for x in chain])
        # Get available strikes
        available_strikes = sorted(set([contract.Strike for contract in chain]))
        # Determine possible symmetric wing spreads
        distances = [strike - atm_strike for strike in available_strikes]
        positive_distances = sorted(set([d for d in distances if d > 0]))
        possible_wing_spreads = [d for d in positive_distances if -d in distances]
        p_deg_freedom, p_mean, p_scale = self.create_t_distribution(chain.underlying.price)

        if not possible_wing_spreads:
            self.Debug("No symmetric wing spreads available.")
            return None, None

        num_spreads = min(15, len(possible_wing_spreads))
        if num_spreads < 1:
            self.Debug("Not enough wing spreads available.")
            return None, None

        # Evenly select wing spreads from the possible ones
        indices = np.linspace(0, len(possible_wing_spreads) - 1, num_spreads).astype(int)
        wing_spreads = [possible_wing_spreads[i] for i in indices]
        best_ev = float('-inf')
        best_position = None

        for wing_spread in wing_spreads:
            target_put_strike = atm_strike - wing_spread
            target_call_strike = atm_strike + wing_spread
            # Find the closest available strikes to the target strikes
            put_strikes = [s for s in available_strikes if s <= target_put_strike]
            call_strikes = [s for s in available_strikes if s >= target_call_strike]
            if not put_strikes or not call_strikes:
                continue
            closest_put = max(put_strikes)
            closest_call = min(call_strikes)
            # Ensure symmetry
            if abs(closest_put - atm_strike) != abs(closest_call - atm_strike):
                continue
            iron_butterfly = OptionStrategies.IronButterfly(
                self._symbol, closest_put, atm_strike, closest_call, expiry
            )
            position = self.compute_iron_butterfly_metrics(iron_butterfly, chain, atm_strike)
            if not position:
                continue
            curr_ev = self.calculate_EV(p_deg_freedom, p_mean, p_scale, position.expiry_payoff)
            if curr_ev > best_ev:
                best_ev = curr_ev
                best_position = position

        if best_position and best_ev > 0:
            return best_position, best_ev
        else:
            return None, None

    class Position:
        def __init__(self, iron_butterfly, chain, net_premium_received_per_underlying_asset, max_profit, max_loss, m, fee,
                     K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM, atm_strike, entry_time, stop_loss_ratio):
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
            self.stop_loss = self.stop_loss_ratio*(self.max_loss - self.fee) + self.fee # 50% stop loss
            # Calculate price bounds for stop loss
            self.lower_bound, self.upper_bound = self.calculate_price_bounds(-self.stop_loss)
            self.lower_break_even, self.upper_break_even = self.calculate_price_bounds(0)

        def curr_payoff(self, iron_butterfly_prem_per_underlying_asset):
            # Original payoff calculation
            P_T = ((self.net_premium_received_per_underlying_asset - iron_butterfly_prem_per_underlying_asset) * self.m) - self.fee 
            # Adjust payoff for stop loss
            P_T = np.maximum(P_T, -self.stop_loss) # Negative because loss is negative payoff
            return P_T

        def expiry_payoff(self, x):
            # Payoffs for the OTM and ATM options
            C_T_OTM = np.maximum(x - self.K_C_OTM, 0)
            C_T_ATM = np.maximum(x - self.K_C_ATM, 0)
            P_T_OTM = np.maximum(self.K_P_OTM - x, 0)
            P_T_ATM = np.maximum(self.K_P_ATM - x, 0)
            # Iron butterfly premium per underlying asset
            iron_butterfly_prem_per_underlying_asset = (C_T_ATM + P_T_ATM) - (C_T_OTM + P_T_OTM)
            return self.curr_payoff(iron_butterfly_prem_per_underlying_asset)

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
        # enter True => we are calculating the premium we get from opening position. Else, premium we pay to exit.
        C_ATM = P_ATM = C_OTM = P_OTM = None
        for leg in iron_butterfly.OptionLegs:
            contract = next(
                (x for x in chain if x.Strike == leg.Strike and x.Right == leg.Right and x.Expiry.date() == leg.Expiration.date()),
                None
            )
            if contract is None:
                continue
            if leg.Quantity == -1:
                if enter:
                    option_price = contract.BidPrice
                else:
                    option_price = contract.AskPrice
            elif leg.Quantity == 1: 
                if enter:
                    option_price = contract.AskPrice
                else:
                    option_price = contract.BidPrice

            if leg.Quantity == -1 and leg.Right == OptionRight.Call:
                C_ATM = option_price
            elif leg.Quantity == -1 and leg.Right == OptionRight.Put:
                P_ATM = option_price
            elif leg.Quantity == 1 and leg.Right == OptionRight.Call:
                C_OTM = option_price
            elif leg.Quantity == 1 and leg.Right == OptionRight.Put:
                P_OTM = option_price

        if None in [C_ATM, P_ATM, C_OTM, P_OTM]:
            return None

        return (C_ATM + P_ATM) - (C_OTM + P_OTM)

    def compute_iron_butterfly_metrics(self, iron_butterfly, chain, atm_strike):
        m = 100  # Contract multiplier for options
        fee = 4  # Set fees according to your brokerage model

        K_C_ATM = K_P_ATM = K_C_OTM = K_P_OTM = None
        for leg in iron_butterfly.OptionLegs:
            contract = next((x for x in chain if x.Strike == leg.Strike and x.Right == leg.Right and x.Expiry.date() == leg.Expiration.date()), None)
            if contract is None:
                continue
            if leg.Quantity == -1 and leg.Right == OptionRight.Call:
                K_C_ATM = leg.Strike
            elif leg.Quantity == -1 and leg.Right == OptionRight.Put:
                K_P_ATM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Call:
                K_C_OTM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Put:
                K_P_OTM = leg.Strike

        if None in [K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM]:
            return None

        net_premium_received_per_underlying_asset = self.iron_butterfly_prem_per_underlying_asset(iron_butterfly, chain, enter=True)
        if net_premium_received_per_underlying_asset is None:
            return None

        max_profit = net_premium_received_per_underlying_asset*m - fee
        wing_width = K_C_OTM - K_C_ATM
        max_loss = ((wing_width - net_premium_received_per_underlying_asset) * m) + fee  
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

    def probability_between_bounds(self, lb, ub, atm):
        new_degrees_of_freedom, new_mean, new_scale = self.create_t_distribution(atm)
        distribution = t(df=new_degrees_of_freedom, loc=new_mean, scale=new_scale)
        return distribution.cdf(ub) - distribution.cdf(lb)

    def create_t_distribution(self, atm):
        degrees_of_freedom, mean, scale = self.param_dict[self.Time.hour]
        return degrees_of_freedom, atm*(1+mean), atm*scale

    def calculate_EV(self, new_degrees_of_freedom, new_mean, new_scale, payoff):
        distribution = t(df=new_degrees_of_freedom, loc=new_mean, scale=new_scale)
        lb = 0.001
        ub = 1 - lb
        x = np.linspace(distribution.ppf(lb), distribution.ppf(ub), 1000)
        pdf = distribution.pdf(x)
        EV = np.sum(payoff(x) * pdf * (x[1] - x[0]))
        return EV

    def OnData(self, slice: Slice) -> None:
        spy_data = slice.Bars.get(self.spy)
        if spy_data:
            self.update_price_window(spy_data.Close)

        historical_volatility = self.calculate_historical_volatility()
        chain = slice.OptionChains.get(self._symbol, None)
        if chain:
            atm_strike = sorted(chain, key=lambda x: abs(x.Strike - chain.Underlying.Price))[0].Strike
            avg_iv = sum(x.ImpliedVolatility for x in chain) / len([i for i in chain])
            if historical_volatility and historical_volatility > 0:
                iv_hv_ratio = avg_iv / historical_volatility
                self.running_window.append(iv_hv_ratio)
        else:
            return

        if self.IsWarmingUp or self.Time.hour in [16]:
            return

        if self.Time.hour == 10:
            self.Liquidate()
            self.position = None
            self.invested = False

        if self.invested:
            if self.position is not None:
                current_price = self.Securities[self.spy].Price
                # Stop loss check
                if current_price <= self.position.lower_bound or current_price >= self.position.upper_bound:
                    self.Debug(f"Closing position entered at {self.position.entry_time} because of stop loss.")
                    self.Liquidate()
                    self.position = None
                    self.invested = False
                    self.no_trades = None
                    pnl = self.Portfolio.TotalPortfolioValue - self.capital_before_investment
                    self.Debug(f"return is {pnl}") 
                    return

                # Profit target check
                # Calculate current profit if we exit now
                iron_butterfly_prem_per_underlying_asset = self.iron_butterfly_prem_per_underlying_asset(self.position.iron_butterfly, chain, enter=False)
                if iron_butterfly_prem_per_underlying_asset is None:
                    self.Debug("Error in calculating current payoff.")
                    return
                
                exit_profit = self.position.curr_payoff(iron_butterfly_prem_per_underlying_asset)

                # Check if exit profit meets 25% profit target
                if exit_profit >= 0.25 * self.position.max_profit:
                    self.Debug(f"Closing position entered at {self.position.entry_time} to lock in a 25% profit target.")
                    self.Liquidate()
                    self.position = None
                    self.invested = False
                    self.no_trades = None
                    pnl = self.Portfolio.TotalPortfolioValue - self.capital_before_investment
                    self.profit_from_exit += pnl
            return

        iv_hv_ratio = avg_iv / historical_volatility
        iv_hv_lower_bound = 75
        percentile = np.percentile(self.running_window, iv_hv_lower_bound)
        if iv_hv_ratio < percentile:
            return

        position, EV_of_expiry_profit = self.select_best_iron_butterfly(chain, atm_strike)
        if position is None:
            return
        win_chance = self.probability_between_bounds(position.lower_break_even, position.upper_break_even, chain.underlying.price)
        if win_chance < 0.5 or EV_of_expiry_profit < 10:
            return

        # Calculate the adjusted maximum loss based on the position's stop loss
        loss_amount = position.stop_loss

        # Given the EV of expiry and the win probability, solve for the average win amount per winning scenario.
        # The formula is derived as follows:
        # EV_of_expiry_profit = p * (win_amount) - (1 - p) * (loss_amount)
        # Rearranging for win_amount:
        # win_amount = (EV_of_expiry_profit + loss_amount*(1 - p)) / p
        win_amount = (EV_of_expiry_profit + loss_amount * (1 - win_chance)) / win_chance

        # Calculate the Kelly fraction (capital_risked_pct) using a Kelly-like formula:
        # Kelly fraction ≈ ((p * win) - ((1 - p) * loss)) / (win * loss)
        # This gives the fraction of capital that should be risked for an optimal growth strategy.
        capital_risked_pct = ((win_chance * win_amount) - ((1 - win_chance) * loss_amount)) / (win_amount * loss_amount)

        # Normalize and cap the Kelly fraction. We don't want to risk more than 15% of capital.
        capital_risked_pct = max(0, min(capital_risked_pct * self.kelly_norm_factor, 0.15))

        # Determine how many spreads (no_trades) to buy based on the capital_risked_pct and the stop loss.
        # We also cap at a maximum of 100 spreads for additional risk control.
        no_trades = min(100, int((capital_risked_pct * self.Portfolio.TotalPortfolioValue) / position.stop_loss)) + 1

        # If the calculated number of trades is zero or negative (e.g., due to unfavorable conditions), skip the trade.
        if no_trades <= 0:
            return

        self.Debug(f"Win probability: {100*win_chance:.2f}% | EV: {EV_of_expiry_profit:.2f} | "
           f"Capital Risked: {capital_risked_pct*100:.2f}% | Trades: {no_trades}")
        self.Debug(f"Trade placed at {self.Time}")

        # Record the current capital for later PnL calculations
        self.capital_before_investment = self.Portfolio.TotalPortfolioValue

        # Enter the trade by buying the calculated number of iron butterfly spreads
        self.Buy(position.iron_butterfly, no_trades)

        # Mark that we are invested and store relevant details
        self.invested = True
        self.no_trades = no_trades
        self.position = position
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

from mpl_finance import candlestick_ochl as candlestick

style.use('dark_background')

BUY_N_HOLD_COLOR = '#FFFFFF'
BOT_COLOR = '#44a769'
BALANCE_COLOR = '#44a769'
UP_COLOR = '#FFFFFF'
DOWN_COLOR = '#44a769'
VOLUME_CHART_HEIGHT = 0.33


def date2num(date):
    converter = mdates.datestr2num(date)
    return converter


class GraphGenerator:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, assets, currency, granularity, datapoints, df_complete, df_features, variables):
        self.assets = assets
        self.currency = currency
        self.granularity = granularity
        self.datapoints = datapoints
        self.df_complete = df_complete
        self.df_features = df_features
        self.variables = variables
        self.candlestick_width = variables['candlestick_width'].get(
            granularity, 1)

        # ? benchmark strats
        self.buy_and_holds = np.zeros(len(df_complete[assets[0]]['Date']))
        self.net_worths = np.zeros(len(df_complete[assets[0]]['Date']))

        self.buy_and_holds[0] = variables['initial_account_balance']
        self.net_worths[0] = variables['initial_account_balance']

        fig = plt.figure()
        fig.suptitle('T-1000 trading on {}'.format(self.currency), fontsize=18)

        self.price_axs = {}
        self.volume_axs = {}
        rowspan = 3
        colspan = 4  # ! idk
        canvas_x_size = len(assets) * rowspan
        canvas_y_size = len(assets) + 2  # ? + balance and net_worth

        self.net_worth_ax = plt.subplot2grid(
            (canvas_x_size, canvas_y_size), (0, 4), rowspan=4, colspan=1)
        self.net_worth_ax.yaxis.tick_right()

        self.balance_ax = plt.subplot2grid(
            (canvas_x_size, canvas_y_size), (5, 4), rowspan=4, colspan=1)
        self.balance_ax.yaxis.tick_right()

        for index, asset in enumerate(assets):
        #     print(index, asset)
        # quit()
            self.price_axs[asset] = plt.subplot2grid((canvas_x_size, canvas_y_size), (len(
                assets) * index, 0), rowspan=rowspan, colspan=colspan)
            self.volume_axs[asset] = self.price_axs[asset].twinx()
            self.price_axs[asset].yaxis.tick_right()

        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # ? Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worth, buy_and_hold, step_range, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()
        # compute performance
        abs_diff = net_worth - buy_and_hold
        avg = (net_worth + buy_and_hold) / 2
        percentage_diff = abs_diff / avg * 100
        # print performance
        self.net_worth_ax.text(0.95, 0.01, '{0:.2f}%'.format(percentage_diff),
                               verticalalignment='bottom',
                               horizontalalignment='right',
                               transform=self.net_worth_ax.transAxes,
                               color='green' if percentage_diff > 0 else 'red', fontsize=15)
        x = np.arange(2)
        y = [buy_and_hold, net_worth]
        self.net_worth_ax.bar(x, y, color=[BUY_N_HOLD_COLOR, BOT_COLOR])
        self.net_worth_ax.set_title(
            "Net Worth ({})".format(self.currency))
        self.net_worth_ax.set_xticklabels(('oi', 'HODL', 'oi', 'Bot'))

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate("{0:.2f}".format(buy_and_hold),
                                   xy=(0, buy_and_hold),
                                   xytext=(0, buy_and_hold),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")
        self.net_worth_ax.annotate("{0:.2f}".format(net_worth),
                                   xy=(1, net_worth),
                                   xytext=(1, net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

    def _render_balance(self, shares_held, balance):
        # Clear the frame rendered last step
        self.balance_ax.clear()
        # compute performance
        # abs_diff = net_worth - buy_and_hold
        # avg = (net_worth + buy_and_hold) / 2
        # percentage_diff = abs_diff / avg * 100
        # print performance
        # self.net_worth_ax.text(0.95, 0.01, '{0:.2f}%'.format(percentage_diff),
        #                       verticalalignment='bottom', horizontalalignment='right',
        #                       transform=self.net_worth_ax.transAxes,
        #                       color='green' if percentage_diff > 0 else 'red', fontsize=15)
        x = np.arange(4)
        y = [shares_held[asset] for asset in self.assets]
        y.append(balance)
        # y = [shares_held1, shares_held2, shares_held3, balance]
        self.balance_ax.bar(x, y, color=BALANCE_COLOR)
        self.balance_ax.set_title("Balance")
        # labels = self.assets
        labels = ('',)
        for asset in self.assets:
            labels += (asset, )
        labels += (self.currency, )
        # print('labels')
        # print(labels)
        # quit()
        # self.balance_ax.set_xticklabels(
        #     ('', self.first_coin, self.second_coin, self.thrid_coin, self.trade_instrument))
        self.balance_ax.set_xticklabels(labels)

        for index, asset in enumerate(self.assets):
            # print('\n\n\n\n\n')
            # print(index, asset)
            # print(self.assets)
            # print('shares_held[asset]')
            # print(shares_held[asset])
            # print('shares_held[asset]')
            # print('\n\n\n\n\n')
            self.balance_ax.annotate("{0:.3f}".format(shares_held[asset]), xy=(index, shares_held[asset]), xytext=(
                index, shares_held[asset]), bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), color='black', fontsize='small')
        self.balance_ax.annotate("{0:.3f}".format(balance),
                                 xy=(len(self.assets), balance),
                                 xytext=(len(self.assets), balance),
                                 bbox=dict(boxstyle='round',
                                           fc='w', ec='k', lw=1),
                                 color="black",
                                 fontsize="small")

    def _render_price(self, current_step, net_worth, dates, step_range):
        candlesticks = {}
        last_dates = {}
        last_closes = {}
        last_highs = {}
        y_limit = {}
        for index, asset in enumerate(self.assets):
            if index == 0:
                self.price_axs[asset].set_title(
                    'Candlesticks')  # ? this can go out?
            self.price_axs[asset].clear()
            candlesticks[asset] = zip(dates,
                                      self.df_features[asset]['open'].values[step_range],
                                      self.df_features[asset]['close'].values[step_range],
                                      self.df_features[asset]['high'].values[step_range],
                                      self.df_features[asset]['low'].values[step_range])
            candlestick(self.price_axs[asset],
                        candlesticks[asset],
                        width=self.candlestick_width,
                        colorup=UP_COLOR,
                        colordown=DOWN_COLOR)
            last_dates[asset] = date2num(
                self.df_complete[asset]['Date'].values[current_step])
            last_closes[asset] = self.df_features[asset]['close'].values[current_step]
            last_highs[asset] = self.df_features[asset]['high'].values[current_step]
            self.price_axs[asset].annotate(s="{0:.4f}".format(last_closes[asset]), xy=(
                last_dates[asset], last_closes[asset]), xytext=(last_dates[asset], last_highs[asset]), bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), color='black', fontsize='small')
            y_limit[asset] = self.price_axs[asset].get_ylim()
            self.price_axs[asset].set_ylim(y_limit[asset][0] - (
                y_limit[asset][1] - y_limit[asset][0]) * VOLUME_CHART_HEIGHT, y_limit[asset][1])
            self.price_axs[asset].set_ylabel(
                '{}/{}'.format(asset, self.currency))

    def _render_volume(self, current_step, net_worth, dates, step_range):
        self.volume_ax1.clear()
        self.volume_ax2.clear()
        self.volume_ax3.clear()

        volume1 = np.array(self.df1['volumefrom'].values[step_range])
        volume2 = np.array(self.df2['volumefrom'].values[step_range])
        volume3 = np.array(self.df3['volumefrom'].values[step_range])

        pos1 = self.df1['open'].values[step_range] - \
            self.df1['close'].values[step_range] < 0
        neg1 = self.df1['open'].values[step_range] - \
            self.df1['close'].values[step_range] > 0
        pos2 = self.df2['open'].values[step_range] - \
            self.df2['close'].values[step_range] < 0
        neg2 = self.df2['open'].values[step_range] - \
            self.df2['close'].values[step_range] > 0
        pos3 = self.df3['open'].values[step_range] - \
            self.df3['close'].values[step_range] < 0
        neg3 = self.df3['open'].values[step_range] - \
            self.df3['close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax1.bar(dates[pos1],
                            volume1[pos1],
                            color=UP_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')
        self.volume_ax1.bar(dates[neg1],
                            volume1[neg1],
                            color=DOWN_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')
        self.volume_ax2.bar(dates[pos2],
                            volume2[pos2],
                            color=UP_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')
        self.volume_ax2.bar(dates[neg2],
                            volume2[neg2],
                            color=DOWN_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')
        self.volume_ax3.bar(dates[pos3],
                            volume3[pos3],
                            color=UP_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')
        self.volume_ax3.bar(dates[neg3],
                            volume3[neg3],
                            color=DOWN_COLOR,
                            alpha=0.4,
                            width=self.candlestick_width,
                            align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax1.set_ylim(0, max(volume1) / VOLUME_CHART_HEIGHT)
        self.volume_ax2.set_ylim(0, max(volume2) / VOLUME_CHART_HEIGHT)
        self.volume_ax3.set_ylim(0, max(volume3) / VOLUME_CHART_HEIGHT)
        self.volume_ax1.yaxis.set_ticks([])
        self.volume_ax2.yaxis.set_ticks([])
        self.volume_ax3.yaxis.set_ticks([])

    # repeated code, need to refactor
    def _render_trades(self, current_step, trades1, trades2, trades3, step_range):
        for trade in trades1:
            if trade['step'] in step_range:
                date = date2num(self.df1['Date'].values[trade['step']])
                high = self.df1['high'].values[trade['step']]
                low = self.df1['low'].values[trade['step']]
                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                    marker = '^'
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR
                    marker = 'v'
                total = '{0:.5f}'.format(trade['total'])
                self.price_ax1.scatter(
                    date, high_low, color=color, marker=marker, s=50)
                # Print the current price to the price axis
                self.price_ax1.annotate('{} {}'.format(total, self.trade_instrument),
                                        xy=(date, high_low),
                                        xytext=(date, high_low),
                                        color=color,
                                        fontsize=8)

        for trade in trades2:
            if trade['step'] in step_range:
                date = date2num(self.df1['Date'].values[trade['step']])
                high = self.df2['high'].values[trade['step']]
                low = self.df2['low'].values[trade['step']]
                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                    marker = '^'
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR
                    marker = 'v'
                total = '{0:.5f}'.format(trade['total'])
                self.price_ax2.scatter(
                    date, high_low, color=color, marker=marker, s=50)
                # Print the current price to the price axis
                self.price_ax2.annotate('{} {}'.format(total, self.trade_instrument),
                                        xy=(date, high_low),
                                        xytext=(date, high_low),
                                        color=color,
                                        fontsize=8)

        for trade in trades3:
            if trade['step'] in step_range:
                date = date2num(self.df1['Date'].values[trade['step']])
                high = self.df3['high'].values[trade['step']]
                low = self.df3['low'].values[trade['step']]
                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                    marker = '^'
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR
                    marker = 'v'
                total = '{0:.5f}'.format(trade['total'])
                self.price_ax3.scatter(
                    date, high_low, color=color, marker=marker, s=50)
                # Print the current price to the price axis
                self.price_ax3.annotate('{} {}'.format(total, self.trade_instrument),
                                        xy=(date, high_low),
                                        xytext=(date, high_low),
                                        color=color,
                                        fontsize=8)

    def render(self, current_step, net_worth, buy_and_hold, trades, shares_held, balance, window_size):
        # $ troquei trades1..2... por trades
        self.net_worths[current_step] = net_worth
        self.buy_and_holds[current_step] = buy_and_hold

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([date2num(x)
                          for x in self.df_complete[self.assets[0]]['Date'].values[step_range]])

        self._render_net_worth(current_step, net_worth,
                               buy_and_hold, step_range, dates)

        self._render_balance(shares_held, balance)
        self._render_price(current_step, net_worth, dates, step_range)
        # $
        # $ CONTINUAR DAQUI
        # $
        # self._render_volume(current_step, net_worth, dates, step_range)
        # self._render_trades(current_step, trades, step_range)

        # Format the date ticks to be more easily read
        last_asset_index = len(self.assets) - 1
        for index, asset in enumerate(self.assets):
            self.price_axs[asset].set_xticklabels(
                self.df_complete[asset]['Date'].values[step_range], rotation=45, horizontalalignment='right')
        # self.price_ax1.set_xticklabels(self.df1['Date'].values[step_range], rotation=45,
        #                                horizontalalignment='right')
        # self.price_ax2.set_xticklabels(self.df2['Date'].values[step_range], rotation=45,
        #                                horizontalalignment='right')
        # self.price_ax3.set_xticklabels(self.df3['Date'].values[step_range], rotation=45,
        #                                horizontalalignment='right')

        # Hide duplicate net worth date labels
            if not index == last_asset_index:
                plt.setp(
                    self.volume_axs[asset].get_xticklabels(), visible=False)
                plt.setp(
                    self.price_axs[asset].get_xticklabels(), visible=False)

        # plt.setp(self.volume_ax1.get_xticklabels(), visible=False)
        # plt.setp(self.volume_ax2.get_xticklabels(), visible=False)
        # plt.setp(self.price_ax1.get_xticklabels(), visible=False)
        # plt.setp(self.price_ax2.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()

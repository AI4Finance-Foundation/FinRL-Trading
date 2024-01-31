import time, datetime, traceback, sys
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

from loguru import logger
logger.add(f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log", rotation="500 MB")

class FinRlContextinfoHandler():
    """
    用于存储策略运行过程中的上下文数据
    """
    def __init__(self):
        logger.debug('fin_rl_contextinfo_handler初始化')
        self.path = r'C:\QMT实盘\userdata_mini'              # 客户端所在路径
        self.acc_str = 'xxxxxxx'                            # 资金账号
        self.stock_list = ['588000.SH', '588220.SH']        # 存放需要交易的标的列表
        self.order_volume = 1000                            # 存放一次下的订单张数
        self.signal_dict  = {}                               # 存放fin_rl_deal_loigics函数里面的信号

def fin_rl_deal_loigics(data_subscribed):
    """
    策略逻辑处理函数，所有fin_rl产生的信号都在这里处理
    """
    now = datetime.datetime.now()
    # 如果是早上九点40之前，或者下午两点半之后，就不下单
    if now.hour < 9 and now.minute < 40:
        return
    if now.hour > 14:
        if now.hour == 14 and now.minute > 30:
            return
    for stock in data_subscribed:
        try:
            signal = next(k for k, v in finrl_contextinfo_handler.signal_dict.items() if v == 1)
        except StopIteration:
            signal = '信号处理异常'



class FinRlTradercallback(XtQuantTraderCallback):
    """
    交易回调类，用于接收交易推送，同时打印各种有用的信息
    """
    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        logger.critical('连接断开回调')

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:

        # 属性	类型	注释
        # account_type	int	账号类型，参见数据字典
        # account_id	str	资金账号
        # stock_code	str	证券代码，例如"600000.SH"
        # order_id	int	订单编号
        # order_sysid	str	柜台合同编号
        # order_time	int	报单时间
        # order_type	int	委托类型，参见数据字典
        # order_volume	int	委托数量
        # price_type	int	报价类型
        # price	float	委托价格
        # traded_volume	int	成交数量
        # traded_price	float	成交均价
        # order_status	int	委托状态，参见数据字典
        # status_msg	str	委托状态描述，如废单原因
        # strategy_name	str	策略名称
        # order_remark	str	委托备注
        """
        # 上面两句改成logger
        logger.debug(f'委托回调 {order.order_remark()}')
        logger.debug(
            f'oder_id:{order.order_id},order_type,{order.order_type}order_price:{order.order_price},order_volume:{order.order_volume}')

    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        # 属性	类型	注释
        # account_type	int	账号类型，参见数据字典
        # account_id	str	资金账号
        # stock_code	str	证券代码
        # order_type	int	委托类型，参见数据字典
        # traded_id	str	成交编号
        # traded_time	int	成交时间
        # traded_price	float	成交均价
        # traded_volume	int	成交数量
        # traded_amount	float	成交金额
        # order_id	int	订单编号
        # order_sysid	str	柜台合同编号
        # strategy_name	str	策略名称
        # order_remark	str	委托备注

        """
        logger.info(f'成交回调{trade.order_remark}')
        logger.debug(
            f'traded_id:{trade.traded_id},stock_code:{trade.stock_code},traded_price:{trade.traded_price},traded_volume:{trade.traded_volume}')

    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error:XtOrderError 对象
        :return:
        # 属性	类型	注释
        # account_type	int	账号类型，参见数据字典
        # account_id	str	资金账号
        # order_id	int	订单编号
        # error_id	int	下单失败错误码
        # error_msg	str	下单失败具体信息
        # strategy_name	str	策略名称
        # order_remark	str	委托备注
        """
        logger.error(
            f"委托报错回调:报错信息 {order_error.error_msg},错误策略名称{order_error.strategy_name},错误委托备注{order_error.order_remark}")

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        """
        logger.error(f"撤单报错回调")

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        """
        logger.debug(f"异步委托回调 ")
        logger.debug(response)
        logger.debug(f'oder_id:{response.order_id},order_price:{response.order_price},order_qty:{response.order_qty}')

    def on_cancel_order_stock_async_response(self, response):
        """
        :param response: XtCancelOrderResponse 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)


if __name__ == '__main__':
    logger.debug("start")
    # 指定客户端所在路径
    # 创建策略上下文对象，用来存放全局信息
    finrl_contextinfo_handler = FinRlContextinfoHandler()

    # 生成session id 整数类型 同时运行的策略不能重复
    session_id = int(time.time())
    xt_trader = XtQuantTrader(finrl_contextinfo_handler.path, session_id)
    # 开启主动请求接口的专用线程 开启后在on_stock_xxx回调函数里调用XtQuantTrader.query_xxx函数不会卡住回调线程，但是查询和推送的数据在时序上会变得不确定
    # 详见: http://docs.thinktrader.net/vip/pages/ee0e9b/#开启主动请求接口的专用线程
    # xt_trader.set_relaxed_response_order_enabled(True)

    # 创建资金账号为 xxxxxxx 的证券账号对象
    acc = StockAccount(finrl_contextinfo_handler.acc_str, 'STOCK')
    # 创建交易回调类对象，并声明接收回调
    finrl_trader_callback = FinRlTradercallback()
    xt_trader.register_callback(finrl_trader_callback)
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    logger.info(f'建立交易连接，返回0表示连接成功{connect_result}' )

    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(acc)
    logger.info(f'对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功{subscribe_result}')

    # 这一行是注册全推回调函数 包括下单判断 安全起见处于注释状态 确认理解效果后再放开
    xtdata.subscribe_whole_quote(code_list=finrl_contextinfo_handler.stock_list, callback=fin_rl_deal_loigics)
    # 阻塞主线程退出
    xt_trader.run_forever()
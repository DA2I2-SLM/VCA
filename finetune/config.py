from dataclasses import dataclass, field
import os
from typing import List

_SCRATCH = os.path.expandvars(os.environ.get('VCA_SCRATCH', '/scratch/$USER/Kronos'))


CRYPTO_TOP20 = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'UNIUSDT',
    'ETCUSDT', 'XLMUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT',
]


CSI300_PIT_SYMBOLS = [
    'sh600000', 'sh600004', 'sh600008', 'sh600009', 'sh600010', 'sh600011', 'sh600015', 'sh600016',
    'sh600018', 'sh600019', 'sh600021', 'sh600022', 'sh600023', 'sh600025', 'sh600027', 'sh600028',
    'sh600029', 'sh600030', 'sh600031', 'sh600036', 'sh600037', 'sh600038', 'sh600048', 'sh600050',
    'sh600058', 'sh600060', 'sh600061', 'sh600066', 'sh600068', 'sh600074', 'sh600079', 'sh600085',
    'sh600089', 'sh600098', 'sh600100', 'sh600104', 'sh600108', 'sh600109', 'sh600111', 'sh600115',
    'sh600118', 'sh600143', 'sh600150', 'sh600153', 'sh600157', 'sh600166', 'sh600170', 'sh600176',
    'sh600177', 'sh600183', 'sh600188', 'sh600196', 'sh600208', 'sh600219', 'sh600221', 'sh600233',
    'sh600252', 'sh600256', 'sh600267', 'sh600271', 'sh600276', 'sh600277', 'sh600297', 'sh600299',
    'sh600309', 'sh600315', 'sh600316', 'sh600317', 'sh600332', 'sh600339', 'sh600340', 'sh600346',
    'sh600348', 'sh600350', 'sh600352', 'sh600362', 'sh600369', 'sh600372', 'sh600373', 'sh600376',
    'sh600383', 'sh600390', 'sh600395', 'sh600398', 'sh600406', 'sh600415', 'sh600436', 'sh600438',
    'sh600446', 'sh600482', 'sh600485', 'sh600487', 'sh600489', 'sh600497', 'sh600498', 'sh600516',
    'sh600518', 'sh600519', 'sh600522', 'sh600535', 'sh600547', 'sh600549', 'sh600566', 'sh600570',
    'sh600578', 'sh600582', 'sh600583', 'sh600585', 'sh600588', 'sh600597', 'sh600600', 'sh600606',
    'sh600633', 'sh600637', 'sh600642', 'sh600648', 'sh600649', 'sh600655', 'sh600660', 'sh600663',
    'sh600664', 'sh600666', 'sh600674', 'sh600682', 'sh600685', 'sh600688', 'sh600690', 'sh600703',
    'sh600704', 'sh600705', 'sh600717', 'sh600718', 'sh600733', 'sh600737', 'sh600739', 'sh600741',
    'sh600745', 'sh600754', 'sh600760', 'sh600783', 'sh600795', 'sh600804', 'sh600809', 'sh600816',
    'sh600820', 'sh600827', 'sh600837', 'sh600839', 'sh600848', 'sh600863', 'sh600867', 'sh600871',
    'sh600873', 'sh600875', 'sh600880', 'sh600886', 'sh600887', 'sh600893', 'sh600895', 'sh600900',
    'sh600909', 'sh600919', 'sh600926', 'sh600928', 'sh600958', 'sh600959', 'sh600968', 'sh600977',
    'sh600989', 'sh600998', 'sh600999', 'sh601006', 'sh601009', 'sh601012', 'sh601016', 'sh601018',
    'sh601021', 'sh601066', 'sh601077', 'sh601088', 'sh601098', 'sh601099', 'sh601100', 'sh601106',
    'sh601108', 'sh601111', 'sh601117', 'sh601118', 'sh601127', 'sh601138', 'sh601155', 'sh601158',
    'sh601162', 'sh601163', 'sh601166', 'sh601168', 'sh601169', 'sh601179', 'sh601186', 'sh601198',
    'sh601211', 'sh601212', 'sh601216', 'sh601225', 'sh601228', 'sh601229', 'sh601231', 'sh601236',
    'sh601238', 'sh601258', 'sh601288', 'sh601298', 'sh601318', 'sh601319', 'sh601328', 'sh601333',
    'sh601336', 'sh601360', 'sh601375', 'sh601377', 'sh601390', 'sh601398', 'sh601555', 'sh601577',
    'sh601600', 'sh601601', 'sh601607', 'sh601608', 'sh601611', 'sh601618', 'sh601628', 'sh601633',
    'sh601658', 'sh601668', 'sh601669', 'sh601688', 'sh601698', 'sh601699', 'sh601718', 'sh601727',
    'sh601766', 'sh601788', 'sh601800', 'sh601808', 'sh601816', 'sh601818', 'sh601828', 'sh601838',
    'sh601857', 'sh601866', 'sh601872', 'sh601877', 'sh601878', 'sh601881', 'sh601888', 'sh601898',
    'sh601899', 'sh601901', 'sh601916', 'sh601919', 'sh601928', 'sh601929', 'sh601933', 'sh601939',
    'sh601958', 'sh601966', 'sh601969', 'sh601985', 'sh601988', 'sh601989', 'sh601991', 'sh601992',
    'sh601997', 'sh601998', 'sh603000', 'sh603019', 'sh603156', 'sh603160', 'sh603259', 'sh603260',
    'sh603288', 'sh603369', 'sh603501', 'sh603658', 'sh603699', 'sh603799', 'sh603833', 'sh603858',
    'sh603885', 'sh603899', 'sh603986', 'sh603993', 'sz000001', 'sz000002', 'sz000008', 'sz000009',
    'sz000027', 'sz000039', 'sz000046', 'sz000060', 'sz000061', 'sz000063', 'sz000066', 'sz000069',
    'sz000100', 'sz000156', 'sz000157', 'sz000166', 'sz000333', 'sz000338', 'sz000400', 'sz000401',
    'sz000402', 'sz000408', 'sz000413', 'sz000415', 'sz000423', 'sz000425', 'sz000503', 'sz000536',
    'sz000538', 'sz000539', 'sz000540', 'sz000553', 'sz000555', 'sz000559', 'sz000568', 'sz000581',
    'sz000596', 'sz000598', 'sz000623', 'sz000625', 'sz000627', 'sz000629', 'sz000630', 'sz000651',
    'sz000656', 'sz000661', 'sz000671', 'sz000686', 'sz000703', 'sz000708', 'sz000709', 'sz000712',
    'sz000718', 'sz000723', 'sz000725', 'sz000728', 'sz000729', 'sz000738', 'sz000750', 'sz000768',
    'sz000776', 'sz000778', 'sz000783', 'sz000786', 'sz000792', 'sz000793', 'sz000800', 'sz000825',
    'sz000826', 'sz000831', 'sz000839', 'sz000858', 'sz000860', 'sz000869', 'sz000876', 'sz000878',
    'sz000883', 'sz000895', 'sz000898', 'sz000917', 'sz000937', 'sz000938', 'sz000959', 'sz000960',
    'sz000961', 'sz000963', 'sz000970', 'sz000977', 'sz000983', 'sz000999', 'sz001965', 'sz001979',
    'sz002001', 'sz002007', 'sz002008', 'sz002010', 'sz002024', 'sz002027', 'sz002032', 'sz002038',
    'sz002044', 'sz002049', 'sz002050', 'sz002051', 'sz002065', 'sz002074', 'sz002081', 'sz002085',
    'sz002120', 'sz002129', 'sz002131', 'sz002142', 'sz002146', 'sz002152', 'sz002153', 'sz002157',
    'sz002174', 'sz002179', 'sz002183', 'sz002195', 'sz002202', 'sz002230', 'sz002236', 'sz002241',
    'sz002252', 'sz002271', 'sz002292', 'sz002294', 'sz002299', 'sz002304', 'sz002310', 'sz002311',
    'sz002344', 'sz002352', 'sz002353', 'sz002371', 'sz002375', 'sz002385', 'sz002399', 'sz002400',
    'sz002410', 'sz002411', 'sz002415', 'sz002416', 'sz002422', 'sz002424', 'sz002426', 'sz002429',
    'sz002450', 'sz002456', 'sz002460', 'sz002463', 'sz002465', 'sz002466', 'sz002468', 'sz002470',
    'sz002475', 'sz002493', 'sz002500', 'sz002508', 'sz002555', 'sz002558', 'sz002568', 'sz002570',
    'sz002572', 'sz002594', 'sz002601', 'sz002602', 'sz002603', 'sz002607', 'sz002608', 'sz002624',
    'sz002625', 'sz002653', 'sz002673', 'sz002714', 'sz002736', 'sz002739', 'sz002773', 'sz002797',
    'sz002831', 'sz002839', 'sz002841', 'sz002916', 'sz002925', 'sz002938', 'sz002939', 'sz002945',
    'sz002958', 'sz003816', 'sz300002', 'sz300003', 'sz300014', 'sz300015', 'sz300017', 'sz300024',
    'sz300027', 'sz300033', 'sz300058', 'sz300059', 'sz300070', 'sz300072', 'sz300085', 'sz300104',
    'sz300122', 'sz300124', 'sz300133', 'sz300136', 'sz300142', 'sz300144', 'sz300146', 'sz300168',
    'sz300182', 'sz300251', 'sz300296', 'sz300315', 'sz300347', 'sz300408', 'sz300413', 'sz300433',
    'sz300498', 'sz300601', 'sz300628',
]


CSI500_PIT_SYMBOLS = [
    'sh600004', 'sh600006', 'sh600008', 'sh600017', 'sh600021', 'sh600022', 'sh600026', 'sh600037',
    'sh600039', 'sh600053', 'sh600056', 'sh600058', 'sh600059', 'sh600060', 'sh600062', 'sh600064',
    'sh600067', 'sh600073', 'sh600074', 'sh600078', 'sh600079', 'sh600086', 'sh600088', 'sh600094',
    'sh600098', 'sh600108', 'sh600110', 'sh600112', 'sh600120', 'sh600122', 'sh600125', 'sh600126',
    'sh600132', 'sh600138', 'sh600141', 'sh600143', 'sh600150', 'sh600151', 'sh600153', 'sh600155',
    'sh600158', 'sh600160', 'sh600161', 'sh600162', 'sh600166', 'sh600167', 'sh600169', 'sh600171',
    'sh600175', 'sh600176', 'sh600179', 'sh600180', 'sh600183', 'sh600184', 'sh600187', 'sh600195',
    'sh600197', 'sh600198', 'sh600199', 'sh600200', 'sh600201', 'sh600216', 'sh600219', 'sh600220',
    'sh600236', 'sh600239', 'sh600240', 'sh600251', 'sh600256', 'sh600258', 'sh600259', 'sh600260',
    'sh600261', 'sh600266', 'sh600267', 'sh600270', 'sh600273', 'sh600277', 'sh600280', 'sh600282',
    'sh600284', 'sh600289', 'sh600291', 'sh600292', 'sh600298', 'sh600300', 'sh600307', 'sh600312',
    'sh600315', 'sh600316', 'sh600317', 'sh600320', 'sh600325', 'sh600329', 'sh600333', 'sh600335',
    'sh600337', 'sh600338', 'sh600339', 'sh600346', 'sh600348', 'sh600350', 'sh600351', 'sh600366',
    'sh600373', 'sh600376', 'sh600380', 'sh600387', 'sh600388', 'sh600389', 'sh600392', 'sh600393',
    'sh600395', 'sh600397', 'sh600403', 'sh600409', 'sh600410', 'sh600415', 'sh600416', 'sh600418',
    'sh600422', 'sh600425', 'sh600426', 'sh600428', 'sh600429', 'sh600432', 'sh600433', 'sh600435',
    'sh600436', 'sh600438', 'sh600439', 'sh600446', 'sh600456', 'sh600458', 'sh600460', 'sh600466',
    'sh600467', 'sh600468', 'sh600478', 'sh600481', 'sh600482', 'sh600483', 'sh600486', 'sh600487',
    'sh600488', 'sh600490', 'sh600496', 'sh600497', 'sh600498', 'sh600499', 'sh600500', 'sh600503',
    'sh600507', 'sh600509', 'sh600510', 'sh600511', 'sh600515', 'sh600516', 'sh600517', 'sh600521',
    'sh600522', 'sh600525', 'sh600528', 'sh600529', 'sh600535', 'sh600536', 'sh600537', 'sh600545',
    'sh600546', 'sh600549', 'sh600551', 'sh600557', 'sh600563', 'sh600565', 'sh600566', 'sh600567',
    'sh600572', 'sh600575', 'sh600578', 'sh600580', 'sh600582', 'sh600584', 'sh600586', 'sh600587',
    'sh600594', 'sh600596', 'sh600597', 'sh600598', 'sh600600', 'sh600601', 'sh600611', 'sh600612',
    'sh600614', 'sh600616', 'sh600617', 'sh600618', 'sh600623', 'sh600628', 'sh600633', 'sh600635',
    'sh600636', 'sh600639', 'sh600640', 'sh600642', 'sh600643', 'sh600645', 'sh600648', 'sh600649',
    'sh600651', 'sh600654', 'sh600655', 'sh600657', 'sh600664', 'sh600667', 'sh600673', 'sh600675',
    'sh600682', 'sh600685', 'sh600687', 'sh600694', 'sh600699', 'sh600702', 'sh600704', 'sh600707',
    'sh600717', 'sh600718', 'sh600720', 'sh600728', 'sh600729', 'sh600733', 'sh600736', 'sh600737',
    'sh600739', 'sh600740', 'sh600743', 'sh600745', 'sh600748', 'sh600750', 'sh600751', 'sh600754',
    'sh600755', 'sh600757', 'sh600759', 'sh600760', 'sh600761', 'sh600763', 'sh600765', 'sh600770',
    'sh600773', 'sh600776', 'sh600777', 'sh600779', 'sh600780', 'sh600782', 'sh600787', 'sh600790',
    'sh600797', 'sh600801', 'sh600803', 'sh600804', 'sh600805', 'sh600808', 'sh600809', 'sh600811',
    'sh600812', 'sh600816', 'sh600820', 'sh600823', 'sh600825', 'sh600826', 'sh600827', 'sh600830',
    'sh600831', 'sh600835', 'sh600839', 'sh600844', 'sh600845', 'sh600848', 'sh600850', 'sh600851',
    'sh600859', 'sh600862', 'sh600863', 'sh600864', 'sh600869', 'sh600872', 'sh600874', 'sh600875',
    'sh600879', 'sh600880', 'sh600881', 'sh600884', 'sh600885', 'sh600894', 'sh600895', 'sh600901',
    'sh600903', 'sh600908', 'sh600909', 'sh600917', 'sh600936', 'sh600939', 'sh600959', 'sh600967',
    'sh600970', 'sh600971', 'sh600978', 'sh600981', 'sh600983', 'sh600985', 'sh600993', 'sh600996',
    'sh600997', 'sh601000', 'sh601001', 'sh601002', 'sh601003', 'sh601005', 'sh601010', 'sh601011',
    'sh601012', 'sh601016', 'sh601019', 'sh601020', 'sh601068', 'sh601098', 'sh601099', 'sh601100',
    'sh601101', 'sh601106', 'sh601118', 'sh601126', 'sh601127', 'sh601128', 'sh601139', 'sh601155',
    'sh601168', 'sh601179', 'sh601200', 'sh601208', 'sh601226', 'sh601228', 'sh601231', 'sh601233',
    'sh601311', 'sh601326', 'sh601333', 'sh601369', 'sh601388', 'sh601512', 'sh601515', 'sh601519',
    'sh601566', 'sh601598', 'sh601608', 'sh601611', 'sh601615', 'sh601666', 'sh601678', 'sh601689',
    'sh601699', 'sh601717', 'sh601718', 'sh601777', 'sh601799', 'sh601801', 'sh601811', 'sh601860',
    'sh601865', 'sh601866', 'sh601869', 'sh601872', 'sh601880', 'sh601886', 'sh601908', 'sh601928',
    'sh601929', 'sh601958', 'sh601965', 'sh601966', 'sh601969', 'sh601975', 'sh601990', 'sh601999',
    'sh603000', 'sh603001', 'sh603005', 'sh603019', 'sh603025', 'sh603056', 'sh603077', 'sh603169',
    'sh603188', 'sh603198', 'sh603225', 'sh603228', 'sh603233', 'sh603256', 'sh603317', 'sh603328',
    'sh603338', 'sh603355', 'sh603366', 'sh603369', 'sh603377', 'sh603379', 'sh603444', 'sh603486',
    'sh603501', 'sh603515', 'sh603517', 'sh603528', 'sh603555', 'sh603556', 'sh603567', 'sh603568',
    'sh603569', 'sh603589', 'sh603605', 'sh603650', 'sh603658', 'sh603659', 'sh603698', 'sh603699',
    'sh603707', 'sh603712', 'sh603766', 'sh603786', 'sh603799', 'sh603806', 'sh603816', 'sh603858',
    'sh603866', 'sh603868', 'sh603877', 'sh603882', 'sh603883', 'sh603885', 'sh603888', 'sh603899',
    'sh603939', 'sh603983', 'sz000006', 'sz000008', 'sz000009', 'sz000012', 'sz000021', 'sz000025',
    'sz000027', 'sz000028', 'sz000030', 'sz000031', 'sz000039', 'sz000046', 'sz000049', 'sz000050',
    'sz000060', 'sz000061', 'sz000062', 'sz000066', 'sz000078', 'sz000088', 'sz000089', 'sz000090',
    'sz000099', 'sz000156', 'sz000158', 'sz000301', 'sz000400', 'sz000401', 'sz000402', 'sz000415',
    'sz000417', 'sz000422', 'sz000426', 'sz000488', 'sz000501', 'sz000511', 'sz000513', 'sz000517',
    'sz000519', 'sz000525', 'sz000528', 'sz000536', 'sz000537', 'sz000540', 'sz000541', 'sz000543',
    'sz000547', 'sz000550', 'sz000552', 'sz000553', 'sz000555', 'sz000559', 'sz000563', 'sz000564',
    'sz000566', 'sz000572', 'sz000581', 'sz000582', 'sz000587', 'sz000592', 'sz000596', 'sz000598',
    'sz000600', 'sz000603', 'sz000612', 'sz000616', 'sz000620', 'sz000623', 'sz000629', 'sz000630',
    'sz000631', 'sz000636', 'sz000650', 'sz000652', 'sz000656', 'sz000661', 'sz000662', 'sz000667',
    'sz000669', 'sz000671', 'sz000680', 'sz000681', 'sz000685', 'sz000686', 'sz000688', 'sz000690',
    'sz000697', 'sz000703', 'sz000712', 'sz000717', 'sz000718', 'sz000719', 'sz000723', 'sz000726',
    'sz000727', 'sz000729', 'sz000732', 'sz000735', 'sz000738', 'sz000739', 'sz000750', 'sz000758',
    'sz000761', 'sz000762', 'sz000766', 'sz000777', 'sz000778', 'sz000780', 'sz000785', 'sz000786',
    'sz000788', 'sz000800', 'sz000806', 'sz000807', 'sz000809', 'sz000810', 'sz000813', 'sz000816',
    'sz000823', 'sz000825', 'sz000826', 'sz000829', 'sz000830', 'sz000848', 'sz000850', 'sz000852',
    'sz000860', 'sz000861', 'sz000869', 'sz000877', 'sz000878', 'sz000883', 'sz000886', 'sz000887',
    'sz000897', 'sz000898', 'sz000900', 'sz000919', 'sz000921', 'sz000926', 'sz000927', 'sz000930',
    'sz000931', 'sz000932', 'sz000933', 'sz000937', 'sz000939', 'sz000951', 'sz000959', 'sz000960',
    'sz000961', 'sz000962', 'sz000967', 'sz000969', 'sz000970', 'sz000973', 'sz000975', 'sz000977',
    'sz000979', 'sz000980', 'sz000983', 'sz000987', 'sz000988', 'sz000990', 'sz000997', 'sz000998',
    'sz000999', 'sz001696', 'sz001872', 'sz001914', 'sz002001', 'sz002002', 'sz002004', 'sz002005',
    'sz002010', 'sz002011', 'sz002013', 'sz002019', 'sz002022', 'sz002025', 'sz002028', 'sz002029',
    'sz002030', 'sz002032', 'sz002038', 'sz002041', 'sz002048', 'sz002049', 'sz002050', 'sz002051',
    'sz002052', 'sz002056', 'sz002063', 'sz002064', 'sz002065', 'sz002069', 'sz002073', 'sz002074',
    'sz002075', 'sz002078', 'sz002080', 'sz002081', 'sz002083', 'sz002085', 'sz002091', 'sz002092',
    'sz002093', 'sz002106', 'sz002110', 'sz002118', 'sz002120', 'sz002122', 'sz002123', 'sz002124',
    'sz002127', 'sz002128', 'sz002129', 'sz002131', 'sz002138', 'sz002140', 'sz002147', 'sz002152',
    'sz002155', 'sz002156', 'sz002157', 'sz002161', 'sz002168', 'sz002174', 'sz002176', 'sz002179',
    'sz002180', 'sz002181', 'sz002183', 'sz002185', 'sz002190', 'sz002191', 'sz002194', 'sz002195',
    'sz002203', 'sz002204', 'sz002212', 'sz002216', 'sz002217', 'sz002219', 'sz002221', 'sz002223',
    'sz002225', 'sz002226', 'sz002233', 'sz002237', 'sz002238', 'sz002240', 'sz002242', 'sz002244',
    'sz002249', 'sz002250', 'sz002251', 'sz002254', 'sz002261', 'sz002266', 'sz002267', 'sz002268',
    'sz002269', 'sz002271', 'sz002273', 'sz002275', 'sz002276', 'sz002277', 'sz002280', 'sz002281',
    'sz002285', 'sz002293', 'sz002294', 'sz002299', 'sz002302', 'sz002308', 'sz002309', 'sz002310',
    'sz002311', 'sz002315', 'sz002317', 'sz002320', 'sz002325', 'sz002327', 'sz002332', 'sz002340',
    'sz002342', 'sz002344', 'sz002345', 'sz002353', 'sz002354', 'sz002358', 'sz002359', 'sz002366',
    'sz002368', 'sz002371', 'sz002372', 'sz002373', 'sz002375', 'sz002382', 'sz002384', 'sz002385',
    'sz002387', 'sz002390', 'sz002392', 'sz002393', 'sz002396', 'sz002399', 'sz002400', 'sz002405',
    'sz002407', 'sz002408', 'sz002410', 'sz002414', 'sz002416', 'sz002419', 'sz002422', 'sz002423',
    'sz002424', 'sz002426', 'sz002428', 'sz002429', 'sz002430', 'sz002431', 'sz002434', 'sz002437',
    'sz002439', 'sz002440', 'sz002444', 'sz002458', 'sz002460', 'sz002461', 'sz002463', 'sz002465',
    'sz002466', 'sz002468', 'sz002470', 'sz002479', 'sz002480', 'sz002482', 'sz002489', 'sz002490',
    'sz002491', 'sz002498', 'sz002500', 'sz002503', 'sz002505', 'sz002506', 'sz002507', 'sz002508',
    'sz002509', 'sz002511', 'sz002512', 'sz002517', 'sz002544', 'sz002551', 'sz002557', 'sz002563',
    'sz002568', 'sz002572', 'sz002573', 'sz002574', 'sz002581', 'sz002583', 'sz002588', 'sz002589',
    'sz002595', 'sz002600', 'sz002601', 'sz002602', 'sz002603', 'sz002607', 'sz002612', 'sz002624',
    'sz002625', 'sz002635', 'sz002640', 'sz002642', 'sz002646', 'sz002648', 'sz002653', 'sz002657',
    'sz002662', 'sz002663', 'sz002665', 'sz002670', 'sz002672', 'sz002678', 'sz002681', 'sz002690',
    'sz002698', 'sz002699', 'sz002701', 'sz002705', 'sz002707', 'sz002709', 'sz002714', 'sz002727',
    'sz002745', 'sz002797', 'sz002807', 'sz002812', 'sz002815', 'sz002818', 'sz002821', 'sz002831',
    'sz002839', 'sz002841', 'sz002867', 'sz002901', 'sz002916', 'sz002920', 'sz002925', 'sz002926',
    'sz002936', 'sz002941', 'sz002946', 'sz002948', 'sz002957', 'sz300001', 'sz300002', 'sz300009',
    'sz300010', 'sz300012', 'sz300014', 'sz300017', 'sz300024', 'sz300026', 'sz300027', 'sz300032',
    'sz300033', 'sz300039', 'sz300043', 'sz300055', 'sz300058', 'sz300059', 'sz300070', 'sz300072',
    'sz300085', 'sz300088', 'sz300113', 'sz300115', 'sz300116', 'sz300122', 'sz300133', 'sz300134',
    'sz300136', 'sz300144', 'sz300146', 'sz300147', 'sz300156', 'sz300159', 'sz300166', 'sz300168',
    'sz300180', 'sz300182', 'sz300197', 'sz300199', 'sz300202', 'sz300207', 'sz300212', 'sz300244',
    'sz300251', 'sz300253', 'sz300257', 'sz300266', 'sz300267', 'sz300271', 'sz300273', 'sz300274',
    'sz300285', 'sz300287', 'sz300291', 'sz300296', 'sz300297', 'sz300308', 'sz300315', 'sz300316',
    'sz300324', 'sz300347', 'sz300357', 'sz300376', 'sz300383', 'sz300413', 'sz300418', 'sz300450',
    'sz300459', 'sz300474', 'sz300482', 'sz300496', 'sz300529', 'sz300558', 'sz300595', 'sz300618',
    'sz300630',
]


@dataclass
class Phase2Config:
    data_dir: str = field(default_factory=lambda: f'{_SCRATCH}/data')
    frequency: str = '15m'
    symbols: List[str] = field(default_factory=lambda: list(CRYPTO_TOP20))
    lookback: int = 160
    pred_len: int = 32
    stride: int = 32
    max_context: int = 512
    clip: float = 5.0
    zero_vol_amount: bool = True

    train_start: str = '2021-01-01'
    train_end: str = '2023-12-31'
    val_start: str = '2024-01-01'
    val_end: str = '2024-06-30'
    val2_start: str = ''
    val2_end: str = ''

    data_fraction: float = 1.0
    data_select: str = None
    acf_select_lags: int = 3
    adaptive_k: bool = False
    k_max: int = 10
    adaptive_prior_tau: float = 0.0
    adaptive_mode: str = 'discrepancy'
    over_penalty: float = 1.0
    acf_pooled: bool = False

    arm: str = 'A1'

    rollout_mode: str = 'teacher_forced'

    ar_val_select: bool = False

    save_all_epochs: bool = False

    gumbel_hard: bool = False

    epochs: int = 10
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 3.0
    seed: int = 42
    num_workers: int = 2

    lambda_acf: float = 10.0
    lambda_schedule: str = 'const'
    lambda_mse: float = 0.0
    lambda_lev: float = 0.0
    lambda_kurt: float = 0.0
    lambda_var: float = 0.0
    lambda_dir: float = 0.0
    dir_mode: str = 'add'
    acf_loss_agg: bool = False
    adaptive_lambda: bool = False
    val_select_agg: bool = False
    val_stability_gate: bool = False
    val_ar_max_batches: int = 100000
    k_train: int = 1
    tau_lag: float = 2.0
    gumbel_tau_start: float = 0.5
    gumbel_tau_end: float = 0.1
    p_max: float = 0.5
    warmup_frac: float = 0.2

    alpha_grad: float = 0.07
    grad_norm_interval: int = 10
    ema_lambda_beta: float = 0.98

    rankic_baseline: float = 0.049
    rankic_drop_tol: float = 0.10
    parkinson_r2_floor: float = 0.25

    tokenizer_path: str = 'NeoQuasar/Kronos-Tokenizer-base'
    predictor_path: str = 'NeoQuasar/Kronos-base'

    use_wandb: bool = False
    save_dir: str = field(default_factory=lambda: f'{_SCRATCH}/outputs/phase2')
    run_name: str = 'A1_full'
    log_interval: int = 50


_DATASETS = {
    'CRYPTO': dict(),
    'CSI300_PIT': dict(data_dir=f'{_SCRATCH}/data_csi300_piT', frequency='1d',
                       symbols=list(CSI300_PIT_SYMBOLS),
                       train_start='2015-01-01', train_end='2017-12-31',
                       val_start='2018-01-01', val_end='2018-12-31'),
    'CSI500PIT': dict(data_dir=f'{_SCRATCH}/data_csi500_piT', frequency='1d',
                      symbols=list(CSI500_PIT_SYMBOLS),
                      train_start='2015-01-01', train_end='2017-12-31',
                      val_start='2018-01-01', val_end='2018-12-31'),
}

_A1_BATCH = {8: 8, 16: 16, 32: 64, 48: 8}
_A1_BATCH_CSI = {**_A1_BATCH, 48: 16}
_AR_BATCH = {8: 8, 16: 8, 32: 8, 48: 4}


def _paper_presets() -> dict:
    """The 3 datasets x 3 methods x 4 horizons the paper reports.

    Names follow `{DATASET}_{METHOD}_H{H}`. METHOD is the code arm, not the
    paper's name: `A1` is CE, `MSE` is MSE-AR, `A2_L10` is VCA (lambda_acf=10).
    Every cell pins lr=5e-5, 10 epochs, lookback W=160; only the AR batch size
    shrinks with H, because the differentiable rollout holds the autograd graph
    across all H steps.

    Crypto uses stride=H (disjoint windows -- 15-minute bars are plentiful);
    CSI uses stride=8 (overlapping, since daily A-share history is far shorter).
    """
    out = {}
    for ds, data in _DATASETS.items():
        stride_csi = ds != 'CRYPTO'
        for H in (8, 16, 32, 48):
            stride = 8 if stride_csi else H
            common = dict(data_fraction=1.0, lr=5e-5, epochs=10,
                          pred_len=H, lookback=160, stride=stride, **data)
            ar = dict(rollout_mode='full_ar', batch_size=_AR_BATCH[H],
                      ar_val_select=True, gumbel_hard=True)
            lo = f'{ds.lower()}_{{}}_h{H}'
            out[f'{ds}_A1_H{H}'] = dict(
                arm='A1', run_name=lo.format('a1'),
                batch_size=(_A1_BATCH_CSI if stride_csi else _A1_BATCH)[H],
                **common)
            out[f'{ds}_MSE_H{H}'] = dict(
                arm='A1-MSE', run_name=lo.format('mse'),
                lambda_acf=0.0, lambda_mse=200.0, **ar, **common)
            out[f'{ds}_A2_L10_H{H}'] = dict(
                arm='A2', run_name=lo.format('a2_l10'),
                lambda_acf=10.0, k_train=1, acf_loss_agg=True, **ar, **common)
    return out


def arm_preset(arm: str = None, **overrides) -> Phase2Config:
    """Build the config for `arm`. Called with no argument it returns the
    sorted list of valid arm names instead -- that is what the CLI uses for
    its --arm choices, so the two can no longer drift apart.

    Prefer the `{DATASET}_{METHOD}_H{H}` presets: those are the paper's cells.
    The three bare names below are the original pre-paper debug arms, kept as
    aliases so older scripts still run. They are NOT the paper's recipes --
    `A2` in particular is lr=1e-4, teacher-forced, k_train=3, per-window ACF
    target, i.e. none of the four things VCA needs. Use `*_A2_L10_H*` for VCA.
    """
    presets = {
        'A1': dict(lr=1e-4, arm='A1', run_name='A1_full', data_fraction=1.0),
        'A2': dict(lr=1e-4, arm='A2', run_name='A2_full', data_fraction=1.0,
                   k_train=3),
        'A1-MSE': dict(arm='A1-MSE', run_name='A1-MSE_full', data_fraction=1.0,
                       rollout_mode='full_ar', batch_size=8, ar_val_select=True,
                       gumbel_hard=True, lambda_acf=0.0, lambda_mse=200.0),
        **_paper_presets(),
    }
    if arm is None:
        return sorted(presets)
    if arm not in presets:
        raise ValueError(f"Unknown arm '{arm}'. Choose from {list(presets)}")
    kwargs = {**presets[arm], **overrides}
    return Phase2Config(**kwargs)

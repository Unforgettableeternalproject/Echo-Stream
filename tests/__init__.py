"""測試套件。

有這個檔案是為了讓測試之間能共用 ``tests.conftest`` 的假 backend——
沒有它時 pytest 不會把 rootdir 放進 sys.path，跨檔 import 會失敗。
"""

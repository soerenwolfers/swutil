'''
Test swutil.logs
'''
import unittest
from swutil.logs import Log

class TestLogs(unittest.TestCase):

    def test_Log(self):
        log=Log()
        log.log(message='Wait')
        log.log(message='Cleanup')
        log.log(group='3',message='whatdup')
        log.print_log()
        print(log)
    

if __name__ == "__main__":
    unittest.main()

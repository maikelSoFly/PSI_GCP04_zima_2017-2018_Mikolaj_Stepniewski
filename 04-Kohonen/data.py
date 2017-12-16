# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-12T19:04:33+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: data.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-16T13:21:04+01:00
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



import requests
from contextlib import closing
import csv
import codecs

class DataReader:
    def __init__(self, url, delimiter=','):
        self.__dict__['_url'] = url
        self.__dict__['_encoding'] = 'utf-8'
        self.__dict__['_delimiter'] = delimiter

    def parse(self):
        with closing(requests.get(self._url, stream=True)) as r:
            reader = csv.reader(
                codecs.iterdecode(r.iter_lines(),self._encoding),
                delimiter=self._delimiter,
                quotechar='"'
            )

            data = []
            for row in reader:
                data.append(row)

        return data

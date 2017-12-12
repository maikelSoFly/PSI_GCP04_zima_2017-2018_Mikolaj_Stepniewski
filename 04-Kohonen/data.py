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

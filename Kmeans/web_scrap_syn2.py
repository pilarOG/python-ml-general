# -*- coding: utf-8 -*-

import urllib
from bs4 import BeautifulSoup as soup
import re
import codecs
import unicodedata
import pandas as pd

'''
outp = codecs.open('noticias-tercera.txt', 'w', encoding='utf-8')

h = open('tercera-titulos.txt', 'r').read().split('\n')
for n in h:
    url = n

    content = urllib.urlopen(url)
    charset = content.headers.getheader('Content-Type')
    if 'charset' in charset.split(';')[-1]:
        data = content.read().decode(charset.split('=')[-1])
        for n in re.findall(r'<p>(.*?)<\/p>', data):
            if '<' not in n:
                i = n.replace('&#8220', '').replace('&#8221', '')
                print type(i)
                outp.writelines(i+'\n')
outp.close()
'''


url = 'http://www.emol.com/Buscador/?query=femicidio'
content = urllib.urlopen(url)
charset = content.headers.getheader('Content-Type')
if 'charset' in charset.split(';')[-1]:
    data = content.read().decode(charset.split('=')[-1])
    print data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:59:34 2024

@author: chloe
"""

import json

return_values = [('a', '9', '3', '17'), ('b', '3', '2', '1')]

def dict_construct(id, x, y, z):
    new_dic = {id : {'properties': {} } }
    values = [{'x': x}, {'y': y}, {'z':z}]
    for val in values:
        new_dic[id]['properties'].update(val)
    return new_dic

a_dict = {'id': {} }
for xx in return_values:
    add_dict = dict_construct(*xx)
    a_dict['id'].update(add_dict)
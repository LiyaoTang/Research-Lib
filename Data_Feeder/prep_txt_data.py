#!/usr/bin/env python
# coding: utf-8
'''
construct front-radar txt file into tfrecord
'''

from optparse import OptionParser
from TFRecord_Constructer import *

# parsing args
parser = OptionParser()
parser.add_option('-d', '--dir', dest='input_dir', help='root dir for data', metavar='FILE')
parser.add_option('-o', '--out', dest='output_file', help='output file of TFRecord', metavar='FILE')
parser.add_option('-s', '--shape', dest='img_shape_2D', default='1024-448', help='image shape 2D', metavar='FILE')
parser.add_option('-r', '--re', dest='re_expr', default='dw_((?!_label).)*\.txt', help='re exp to filter input', metavar='FILE')
parser.add_option('--class_num', dest='class_num', default=3, type="int", help='number of classes', metavar='FILE')
parser.add_option('--no_one_hot', dest='use_one_hot', default=True, action='store_false', help='not use one-hot', metavar='FILE')
parser.add_option('--no_meta', dest='collect_meta', default=True, action='store_false', help='not to collect meta data', metavar='FILE')
parser.add_option('--meta_postfix', dest='meta_postfix', default='_meta', help='postfix for file name of meta data', metavar='FILE')

(options, args) = parser.parse_args()

input_dir = options.input_dir.rstrip('/') + '/'
output_file = options.output_file
img_shape_2D = tuple([int(x) for x in options.img_shape_2D.split('-')]) # default to 1024,448
re_expr = options.re_expr
class_num = options.class_num
use_one_hot = options.use_one_hot
collect_meta = options.collect_meta
meta_postfix = options.meta_postfix

if not output_file:
    output_file = input_dir + 'tfrecord'

print('reading from', input_dir, 'output into', output_file)
constructor = txtData_Constructor(input_dir=input_dir, output_file=output_file, class_num=class_num,
                                  img_shape_2D=img_shape_2D, re_expr=re_expr, use_one_hot=use_one_hot, recursive=True)
constructor.write_into_record()

if collect_meta:
    meta_file = output_file + meta_postfix
    constructor.collect_meta_data(meta_file)
    print('collect meta data into', meta_file)

print('finished')
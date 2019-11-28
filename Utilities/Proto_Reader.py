#!/usr/bin/env python
# coding: utf-8
"""
module: utilities for parsing protobuf without proto-header, able to:
    parse proto txt into tree-like structure
    select specified nodes
    write back to proto txt
        => able to modify batch of prototxt by programing
"""

class Node():
    def __init__(self, name=None, parent=None, comment=None):
        self.parent = parent
        self.key = []
        self.val = []
        self.comment = comment
        self.name = name
        self.format = '%s: %s'  # default format
    
    def add(self, k, v=None):
        if v is None:
            v = Node(k, self)  # sub-node initially named after its key
        if k in self.key:
            idx = self.key.index(k)
            if type(self.val[idx]) != list:
                self.val[idx] = [self.val[idx]]  # convert 'repeated' fields into list
            self.val[idx].append(v)
        else:
            self.key.append(k)
            self.val.append(v)
        return v
    
    def iterator(self, recursive=True):
        h = self
        for k, v in zip(h.key, h.val):
            yield k, v
            if type(v) == Node and recursive:
                yield from v.iterator()
    
    def select_nodes(self, re_expr, recursive=True):
        matched_nodes = []
        for k, v in self.iterator(recursive):
            if bool(re.fullmatch(re_expr, k)):
                if type(v) == list:
                    matched_nodes += [n for n in v if type(n) == Node]
                elif type(v) == Node:
                    matched_nodes.append(v)
        return matched_nodes

    def __getitem__(self, i):  # enable indexing
        assert i in self.key
        return self.val[self.key.index(i)]
    
    def __setitem__(self, k, v):  # enable assignment by index
        if k in self.key:
            self.val[self.key.index(k)] = v
        else:
            self.key.append(k)
            self.val.append(v)
    
    def __repr__(self):
        return str(dict(zip(self.key, self.val)))
    
    def __str__(self):
        return self.to_file()
    
    def copy(self):
        cpy = Node(name=self.name, parent=self.parent, comment=self.comment)
        cpy.key = [k for k in self.key]
        cpy.val = [v.copy() if type(v) == Node else v for v in self.val]
        cpy.format = self.format
        return cpy
    
    def to_file(self, path=None):
        """
        revert back into proto txt
        """
        lines = []
        prefix=''
        def _add_node(h, prefix):
            # lines.append('add node: ' + str(h.name))
            if h.name is not None:  # non-root
                if h.comment is not None:  # if has comment
                    lines.append(prefix + h.comment)
                lines.append(prefix + h.name + ' {')
                prefix += ' '*4
                
            for k, v in h.iterator(recursive=False):
                if type(v) == Node:
                    _add_node(v, prefix)
                else:
                    _add_val(h, k, v, prefix)
            
            if h.name is not None:
                lines.append(prefix[:-4] + '}')
        
        def _add_val(h, k, v, prefix):
            # lines.append('add val: ' + str(k))
            if type(v) == list:
                for n in v:
                    if type(n) == Node:
                        lines.append('')
                        _add_node(n, prefix + ' '*4)
                    else:
                        lines.append(prefix + h.format % (k, n))
            else:
                lines.append(prefix + h.format % (k, v))

        _add_node(self, prefix)
        stream = '\n'.join(lines)

        if path is not None:
            with open(path, 'w') as f:
                f.write(stream)
        return stream

    def read_file(self, path, comment='#'):
        """
        read protobuf, any line containing "#" (default choice) would be treated as comment
        """
        f = open(path, 'r').read()
        f = [l for l in f.split('\n')]
        h = self
        cur_comment = None
        
        for l in f:
            if comment is not None and comment in l:
                cur_comment = l.strip()

            if '}' in l:
                h = h.parent

            elif '{' in l:
                l = l.strip('{').strip()
                h = h.add(l)
                h.comment = cur_comment
                cur_comment = None

            elif ':' in l:
                l = [i.strip() for i in l.split(':')]
                h.add(l[0], l[1])
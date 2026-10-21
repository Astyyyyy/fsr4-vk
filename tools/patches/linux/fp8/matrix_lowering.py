#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Optimize FP8 matrix loads for the tested RADV Wave32 layout.

Matches shader structure rather than IDs or pass names. Validate GPU output
before using these device-specific rewrites.
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

def run(cmd):
    r = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(r.stdout)
    return r.stdout

def definitions(text):
    return {i: (op, a.split()) for i, op, a in re.findall(r"^\s*(%\w+) = (\w+)(?: (.*))?$", text, re.M)}

class Module:
    def __init__(self, text):
        text = re.sub(r'^.*Op(?:Member)?Name .*\n', '', text, flags=re.M)
        self.text = text
        self.d = definitions(text)
        self.globals = []
        self.serial = 0
        self.report = {}
        self.uint = self.type('OpTypeInt', ['32', '0'])
        self.byte = self.type('OpTypeInt', ['8', '0'])
        self.fp8 = self.type('OpTypeFloat', ['8', 'Float8E4M3EXT'])
        self.float = self.type('OpTypeFloat', ['32'])
        self.bool = self.type('OpTypeBool', [])
        vectors = [i for i,(op,a) in self.d.items() if op == 'OpTypeVector' and a == [self.uint, '2']]
        self.vec = vectors[0] if len(vectors) == 1 else None
        self.mats = {i: a for i,(op,a) in self.d.items() if op == 'OpTypeCooperativeMatrixKHR'}
        for i,a in self.mats.items():
            if [self.value(x) for x in a[1:4]] != [3,16,16]:
                raise RuntimeError('Unsupported cooperative matrix scope/shape')
        if not re.search(r'OpExecutionMode %\w+ LocalSize 32 1 1', text):
            raise RuntimeError('Requires LocalSize 32 1 1')
        self.lid = re.findall(r'OpDecorate (%\w+) BuiltIn LocalInvocationId', text)
        if len(self.lid) != 1:
            raise RuntimeError('Requires one LocalInvocationId')
        self.lid = self.lid[0]
        self.lidtype = self.d[self.d[self.lid][1][0]][1][1]

    def fresh(self):
        while True:
            self.serial += 1
            i = f'%waveopt{self.serial}'
            if i not in self.d:
                return i

    def type(self, op, args):
        ids = [i for i,(o,a) in self.d.items() if o == op and a == args]
        if len(ids) != 1:
            raise RuntimeError(f'Expected unique {op} {args}: {ids}')
        return ids[0]

    def value(self, i):
        op,a = self.d.get(i, ('', []))
        if op == 'OpConstant' and self.d.get(a[0], ('',[]))[0] == 'OpTypeInt':
            return int(a[1])
        if op == 'OpCopyObject':
            return self.value(a[1])
        return None

    def constant(self, n, ty=None):
        ty = ty or self.uint
        for i,(op,a) in self.d.items():
            if op == 'OpConstant' and a == [ty,str(n)]:
                return i
        i = self.fresh()
        self.d[i] = ('OpConstant',[ty,str(n)])
        self.globals.append(f'{i} = OpConstant {ty} {n}')
        return i

    def undef(self, ty):
        for i,(op,a) in self.d.items():
            if op == 'OpUndef' and a == [ty]:
                return i
        i=self.fresh()
        self.d[i]=('OpUndef',[ty])
        self.globals.append(f'{i} = OpUndef {ty}')
        return i

    def emit(self, lines, op, ty, *args):
        i=self.fresh()
        argv=[str(a) for a in args]
        self.d[i]=(op,[str(ty),*argv])
        lines.append(f'{i} = {op} {ty} ' + ' '.join(argv))
        return i

    def address(self, lines, ptr, stride):
        op,a=self.d[ptr]
        if op != 'OpAccessChain' or len(a) != 4 or self.value(a[2]) != 0:
            raise RuntimeError('Unsupported matrix pointer chain')
        if self.d[a[0]] != ('OpTypePointer',['StorageBuffer',self.vec]):
            raise RuntimeError('Expected packed StorageBuffer pointer')
        lid=self.emit(lines,'OpLoad',self.lidtype,self.lid)
        lane=self.emit(lines,'OpCompositeExtract',self.uint,lid,0)
        row=self.emit(lines,'OpBitwiseAnd',self.uint,lane,self.constant(15))
        half=self.emit(lines,'OpShiftRightLogical',self.uint,lane,self.constant(4))
        offset=self.emit(lines,'OpIMul',self.uint,row,self.constant(stride))
        offset=self.emit(lines,'OpIAdd',self.uint,offset,half)
        index=self.emit(lines,'OpIAdd',self.uint,a[3],offset)
        return self.emit(lines,'OpAccessChain',a[0],a[1],a[2],index)

    def _storage_class(self, ptr):
        op,a=self.d.get(ptr,('',[]))
        if op not in ['OpAccessChain','OpInBoundsAccessChain'] or not a:
            return None
        top,a0=self.d.get(a[0],('',[]))
        if top!='OpTypePointer' or len(a0)!=2:
            return None
        return a0[0]

    def _root_variables(self, ident, seen=None):
        seen=set() if seen is None else seen
        if ident in seen:
            return set()
        seen.add(ident)
        op,a=self.d.get(ident,('',[]))
        if op=='OpVariable':
            return {ident}
        if op in ['OpAccessChain','OpInBoundsAccessChain'] and len(a)>=2:
            return self._root_variables(a[1],seen)
        if op=='OpBitcast' and len(a)>=2:
            return self._root_variables(a[1],seen)
        return set()

    def _depends_on(self, ident, target, seen=None):
        if ident==target:
            return True
        seen=set() if seen is None else seen
        if ident in seen:
            return False
        seen.add(ident)
        op,a=self.d.get(ident,('',[]))
        if op=='OpVariable' or op=='OpConstant' or op=='OpUndef':
            return False
        if op in ['OpAccessChain','OpInBoundsAccessChain']:
            deps=a[1:]
        elif op=='OpLoad':
            deps=a[1:2]
        elif op in ['OpIAdd','OpISub','OpIMul','OpShiftRightLogical','OpShiftRightArithmetic',
                    'OpShiftLeftLogical','OpBitwiseOr','OpBitwiseXor','OpBitwiseAnd','OpSelect']:
            deps=a[1:]
        elif op in ['OpCopyObject','OpBitcast','OpCompositeExtract']:
            deps=a[1:2]
        else:
            return False
        return any(self._depends_on(x,target,seen) for x in deps if x.startswith('%'))

    def _cloneable(self, ident, seen=None):
        seen=set() if seen is None else seen
        if ident in seen:
            return True
        seen.add(ident)
        op,a=self.d.get(ident,('',[]))
        if op in ['OpConstant','OpUndef']:
            return True
        if op=='OpVariable':
            return len(a)>=2 and a[1]!='Function'
        if op in ['OpAccessChain','OpInBoundsAccessChain']:
            deps=a[1:]
        elif op=='OpLoad':
            deps=a[1:2]
        elif op in ['OpIAdd','OpISub','OpIMul','OpShiftRightLogical','OpShiftRightArithmetic',
                    'OpShiftLeftLogical','OpBitwiseOr','OpBitwiseXor','OpBitwiseAnd','OpSelect']:
            deps=a[1:]
        elif op in ['OpCopyObject','OpBitcast','OpCompositeExtract']:
            deps=a[1:2]
        else:
            return False
        return all((not x.startswith('%')) or self._cloneable(x,seen) for x in deps)

    def _canonical(self, ident, seen=None):
        seen=set() if seen is None else seen
        if ident in seen:
            return ('rec',ident)
        seen.add(ident)
        op,a=self.d.get(ident,('',[]))
        if op=='OpConstant':
            return ('const',a[0],self.value(ident))
        if op=='OpVariable':
            return ('var',ident,a[1] if len(a)>1 else '')
        if op in ['OpAccessChain','OpInBoundsAccessChain']:
            return (op,a[0],*(self._canonical(x,seen.copy()) if x.startswith('%') else ('lit',x) for x in a[1:]))
        if op=='OpLoad':
            return (op,a[0],self._canonical(a[1],seen.copy()),*a[2:])
        if op in ['OpIAdd','OpISub','OpIMul','OpShiftRightLogical','OpShiftRightArithmetic',
                  'OpShiftLeftLogical','OpBitwiseOr','OpBitwiseXor','OpBitwiseAnd','OpSelect']:
            return (op,a[0],*(self._canonical(x,seen.copy()) if x.startswith('%') else ('lit',x) for x in a[1:]))
        if op in ['OpCopyObject','OpBitcast']:
            return (op,a[0],self._canonical(a[1],seen.copy()))
        if op=='OpCompositeExtract':
            return (op,a[0],self._canonical(a[1],seen.copy()),*a[2:])
        return ('id',ident)

    def _clone_value(self, ident, lines, memo):
        if ident in memo:
            return memo[ident]
        op,a=self.d.get(ident,('',[]))
        if op in ['OpConstant','OpUndef']:
            return ident
        if op=='OpVariable':
            if len(a)<2 or a[1]=='Function':
                raise RuntimeError('Cannot clone Function storage value into LDS staging prefix')
            return ident
        if op in ['OpAccessChain','OpInBoundsAccessChain']:
            base=self._clone_value(a[1],lines,memo)
            indices=[self._clone_value(x,lines,memo) if x.startswith('%') else x for x in a[2:]]
            out=self.emit(lines,op,a[0],base,*indices)
        elif op=='OpLoad':
            ptr=self._clone_value(a[1],lines,memo)
            out=self.emit(lines,op,a[0],ptr,*a[2:])
        elif op in ['OpIAdd','OpISub','OpIMul','OpShiftRightLogical','OpShiftRightArithmetic',
                    'OpShiftLeftLogical','OpBitwiseOr','OpBitwiseXor','OpBitwiseAnd']:
            values=[self._clone_value(x,lines,memo) if x.startswith('%') else x for x in a[1:]]
            out=self.emit(lines,op,a[0],*values)
        elif op=='OpSelect':
            values=[self._clone_value(x,lines,memo) if x.startswith('%') else x for x in a[1:]]
            out=self.emit(lines,op,a[0],*values)
        elif op in ['OpCopyObject','OpBitcast']:
            out=self.emit(lines,op,a[0],self._clone_value(a[1],lines,memo))
        elif op=='OpCompositeExtract':
            out=self.emit(lines,op,a[0],self._clone_value(a[1],lines,memo),*a[2:])
        else:
            raise RuntimeError(f'Unsupported LDS staging dependency {op} for {ident}')
        memo[ident]=out
        return out

    def lds_repeated_matrix_cache(self, min_groups=20, min_savings=36):
        """Reuse repeated FP8 matrix loads through LDS.

        Group identical addresses in one straight-line block. Copy each matrix once,
        synchronize the group, then reuse the shared copy.
        """
        wgids=re.findall(r'OpDecorate (%\w+) BuiltIn WorkgroupId',self.text)
        if len(wgids)!=1:
            self.report['lds_cache']={'matched':False,'reason':'requires one WorkgroupId'}
            return
        wgid=wgids[0]

        lines=self.text.splitlines()
        block=None
        records=[]
        for n,line in enumerate(lines):
            lm=re.match(r'\s*(%\w+) = OpLabel\s*$',line)
            if lm:
                block=lm[1]
                continue
            mt=re.match(r'\s*(%\w+) = OpCooperativeMatrixLoadKHR (.*)',line)
            if not mt:
                continue
            result=mt[1]
            a=mt[2].split()
            if len(a)<5:
                continue
            ty,ptr,layout,stride,*flags=a
            mat=self.mats.get(ty)
            if not mat or mat[0]!=self.fp8:
                continue
            role=self.value(mat[4])
            if role not in [0,1] or self.value(layout)!=role:
                continue
            if flags!=['NonPrivatePointer'] or self._storage_class(ptr)!='StorageBuffer':
                continue
            if not self._depends_on(ptr,wgid) or not self._cloneable(ptr):
                continue
            roots=[x for x in self._root_variables(ptr)
                   if self.d.get(x,('',[]))[0]=='OpVariable'
                   and self.d[x][1][1]=='StorageBuffer']
            if len(roots)!=1:
                continue
            step=self.value(stride)
            if step is None or step<2:
                continue
            records.append({
                'line':n,'block':block,'result':result,'type':ty,'ptr':ptr,
                'layout':layout,'stride':stride,'step':step,'flags':flags,
                'role':role,'root':roots[0],
                'key':(ty,self._canonical(ptr),layout,stride,tuple(flags)),
            })

        # Group by straight-line block, source descriptor and cooperative role.
        clusters={}
        for r in records:
            clusters.setdefault((r['block'],r['root'],r['role']),[]).append(r)
        viable=[]
        for key,items in clusters.items():
            grouped={}
            for r in items:
                grouped.setdefault(r['key'],[]).append(r)
            repeated={k:v for k,v in grouped.items() if len(v)>=2}
            savings=sum(len(v)-1 for v in repeated.values())
            if len(repeated)>=min_groups and savings>=min_savings:
                viable.append((savings,len(repeated),key,repeated))
        if not viable:
            self.report['lds_cache']={
                'matched':False,'reason':'no repeated WorkgroupId-dependent FP8 matrix window',
                'min_groups':min_groups,'min_savings':min_savings,
            }
            return
        viable.sort(reverse=True,key=lambda x:(x[0],x[1]))
        if len(viable)>1 and viable[0][:2]==viable[1][:2]:
            raise RuntimeError('Ambiguous LDS cache candidate')
        savings,group_count,(target_block,root,role),repeated=viable[0]
        groups=sorted(repeated.values(),key=lambda v:v[0]['line'])
        target_results={r['result']:slot for slot,g in enumerate(groups) for r in g}
        first=min(r['line'] for g in groups for r in g)
        last=max(r['line'] for g in groups for r in g)

        # Do not reuse a buffer that is written before the last matrix load.
        for n,line in enumerate(lines[:last+1]):
            sm=re.match(r'\s*OpStore (%\w+) ',line)
            cm=re.match(r'\s*OpCooperativeMatrixStoreKHR (%\w+) ',line)
            ptr=(sm or cm)
            if ptr and root in self._root_variables(ptr[1]):
                raise RuntimeError('LDS cache source is written before its last cached load')

        slots=len(groups)
        total_elements=slots*32
        count_id=self.constant(total_elements)
        array_ty=self.fresh()
        self.d[array_ty]=('OpTypeArray',[self.vec,count_id])
        self.globals.append(f'{array_ty} = OpTypeArray {self.vec} {count_id}')
        array_ptr=self.fresh()
        self.d[array_ptr]=('OpTypePointer',['Workgroup',array_ty])
        self.globals.append(f'{array_ptr} = OpTypePointer Workgroup {array_ty}')
        elem_ptrs=[i for i,(op,a) in self.d.items() if op=='OpTypePointer' and a==['Workgroup',self.vec]]
        if len(elem_ptrs)>1:
            raise RuntimeError('Ambiguous Workgroup packed pointer type')
        if elem_ptrs:
            elem_ptr=elem_ptrs[0]
        else:
            elem_ptr=self.fresh()
            self.d[elem_ptr]=('OpTypePointer',['Workgroup',self.vec])
            self.globals.append(f'{elem_ptr} = OpTypePointer Workgroup {self.vec}')
        lds=self.fresh()
        self.d[lds]=('OpVariable',[array_ptr,'Workgroup'])
        self.globals.append(f'{lds} = OpVariable {array_ptr} Workgroup')

        prefix=[]
        lid=self.emit(prefix,'OpLoad',self.lidtype,self.lid)
        lane=self.emit(prefix,'OpCompositeExtract',self.uint,lid,0)
        row=self.emit(prefix,'OpBitwiseAnd',self.uint,lane,self.constant(15))
        half=self.emit(prefix,'OpShiftRightLogical',self.uint,lane,self.constant(4))
        tight=self.emit(prefix,'OpIMul',self.uint,row,self.constant(2))
        tight=self.emit(prefix,'OpIAdd',self.uint,tight,half)
        memo={}
        for slot,g in enumerate(groups):
            r=g[0]
            op,a=self.d[r['ptr']]
            if op not in ['OpAccessChain','OpInBoundsAccessChain'] or len(a)!=4 or self.value(a[2])!=0:
                raise RuntimeError('Unsupported LDS source matrix pointer chain')
            base=self._clone_value(a[1],prefix,memo)
            index=self._clone_value(a[3],prefix,memo)
            offset=self.emit(prefix,'OpIMul',self.uint,row,self.constant(r['step']))
            offset=self.emit(prefix,'OpIAdd',self.uint,offset,half)
            source_index=self.emit(prefix,'OpIAdd',self.uint,index,offset)
            source=self.emit(prefix,op,a[0],base,a[2],source_index)
            packed=self.emit(prefix,'OpLoad',self.vec,source,'Aligned',8)
            lds_index=tight
            if slot:
                lds_index=self.emit(prefix,'OpIAdd',self.uint,tight,self.constant(slot*32))
            dst=self.emit(prefix,'OpAccessChain',elem_ptr,lds,lds_index)
            prefix.append(f'OpStore {dst} {packed} Aligned 8')
        # Synchronize workgroup execution and make shared writes visible.
        prefix.append(f'OpControlBarrier {self.constant(2)} {self.constant(2)} {self.constant(264)}')

        out=[]
        inserted=False
        redirected=0
        for n,line in enumerate(lines):
            if n==first:
                out.extend(prefix)
                inserted=True
            mt=re.match(r'\s*(%\w+) = OpCooperativeMatrixLoadKHR (.*)',line)
            if mt and mt[1] in target_results:
                a=mt[2].split()
                ty,ptr,layout,stride,*flags=a
                slot=target_results[mt[1]]
                base=self.constant(slot*32)
                lptr=self.emit(out,'OpAccessChain',elem_ptr,lds,base)
                out.append(f'{mt[1]} = OpCooperativeMatrixLoadKHR {ty} {lptr} {layout} {self.constant(2)} ' + ' '.join(flags))
                redirected+=1
            else:
                out.append(line)
        if not inserted or redirected!=len(target_results):
            raise RuntimeError('Failed to rewrite complete LDS cache target set')
        self.text='\n'.join(out)+'\n'
        self.report['lds_cache']={
            'matched':True,
            'block':target_block,
            'matrix_role':role,
            'distinct_staged_matrices':slots,
            'redirected_matrix_loads':redirected,
            'global_matrix_loads_saved':savings,
            'lds_bytes':slots*32*8,
            'source_root':root,
            'min_groups':min_groups,
            'min_savings':min_savings,
        }

    def memory(self, loads=True, stores=True):
        count={'loads':0,'stores':0,'retained_loads':0,'retained_stores':0}
        out=[]
        for line in self.text.splitlines():
            mt=re.match(r'\s*(%\w+) = OpCooperativeMatrixLoadKHR (.*)',line)
            st=re.match(r'\s*OpCooperativeMatrixStoreKHR (.*)',line)
            if not mt and not st:
                out.append(line); continue
            a=(mt[2] if mt else st[1]).split()
            if mt:
                ty,ptr,layout,stride,*flags=a
            else:
                ptr,obj,layout,stride,*flags=a
                ty=self.d[obj][1][0]
            mat=self.mats[ty]
            role=self.value(mat[4])
            storage=self._storage_class(ptr)
            eligible=(mat[0]==self.fp8 and role in [0,1] and self.value(layout)==role
                      and storage=='StorageBuffer' and (loads if mt else stores))
            if not eligible:
                count['retained_loads' if mt else 'retained_stores']+=1
                out.append(line); continue
            step=self.value(stride)
            if step is None or step < 2 or flags != ['NonPrivatePointer']:
                raise RuntimeError('Unsupported stride or memory operands')
            lines=[]
            address=self.address(lines,ptr,step)
            if mt:
                packed=self.emit(lines,'OpLoad',self.vec,address,'Aligned',8)
                words=[self.emit(lines,'OpCompositeExtract',self.uint,packed,w) for w in range(2)]
                previous=self.undef(ty)
                for i in range(8):
                    word=words[i//4]
                    if i%4:
                        word=self.emit(lines,'OpShiftRightLogical',self.uint,word,self.constant(i%4*8))
                    byte=self.emit(lines,'OpUConvert',self.byte,word)
                    value=self.emit(lines,'OpBitcast',self.fp8,byte)
                    result=mt[1] if i==7 else self.fresh()
                    lines.append(f'{result} = OpCompositeInsert {ty} {value} {previous} {i}')
                    previous=result
                count['loads']+=1
            else:
                words=[]
                for w in range(2):
                    word=self.constant(0)
                    for b in range(4):
                        value=self.emit(lines,'OpCompositeExtract',self.fp8,obj,w*4+b)
                        value=self.emit(lines,'OpBitcast',self.byte,value)
                        value=self.emit(lines,'OpUConvert',self.uint,value)
                        if b:
                            value=self.emit(lines,'OpShiftLeftLogical',self.uint,value,self.constant(b*8))
                        word=self.emit(lines,'OpBitwiseOr',self.uint,word,value)
                    words.append(word)
                packed=self.emit(lines,'OpCompositeConstruct',self.vec,*words)
                lines.append(f'OpStore {address} {packed} Aligned 8')
                count['stores']+=1
            out.extend(lines)
        self.text='\n'.join(out)+'\n'
        self.report['memory']=count

    def conversions(self):
        targets=re.findall(r'OpDecorate (%\w+) SaturatedToLargestFloat8NormalConversionEXT',self.text)
        functions={}
        for i in targets:
            op,a=self.d[i]
            if op!='OpFConvert' or self.mats[a[0]][0]!=self.fp8:
                raise RuntimeError('Unsupported saturated conversion')
            sop,sa=self.d[a[1]]
            if sop!='OpFunctionCall':
                raise RuntimeError('Saturated conversion must follow recognized helper')
            functions[sa[1]]=True
        ext=re.findall(r'(%\w+) = OpExtInstImport "GLSL.std.450"',self.text)
        if not ext:
            ident = self.fresh()
            pos = self.text.index('OpMemoryModel')
            self.text = self.text[:pos] + f'{ident} = OpExtInstImport "GLSL.std.450"\n' + self.text[pos:]
            ext = [ident]
        if len(ext)!=1:
            raise RuntimeError('Expected GLSL.std.450 import')
        for fn in functions:
            match=re.search(rf'{fn} = OpFunction.*?OpFunctionEnd',self.text,re.S)
            body=match[0]
            expected = Counter({
                'OpFunction': 1, 'OpFunctionParameter': 1, 'OpLabel': 3,
                'OpVariable': 1, 'OpCooperativeMatrixLengthKHR': 1,
                'OpBranch': 1, 'OpPhi': 1, 'OpIAdd': 1,
                'OpInBoundsAccessChain': 2, 'OpLoad': 2, 'OpIsInf': 1,
                'OpSelect': 1, 'OpStore': 1, 'OpULessThan': 1,
                'OpLoopMerge': 1, 'OpBranchConditional': 1,
                'OpReturnValue': 1, 'OpFunctionEnd': 1,
            })
            if Counter(re.findall(r'\bOp\w+', body)) != expected:
                raise RuntimeError('Unknown conversion helper operations')
            # Require the full scalar sequence in the fixup helper.
            pattern=r'(%\w+) = OpIsInf (%\w+) (%\w+)\s+(%\w+) = OpSelect (%\w+) \1 (%\w+) \3'
            repair=re.search(pattern,body)
            if not repair or body.count('OpIsInf')!=1 or body.count('OpSelect')!=1:
                raise RuntimeError('Unknown conversion helper semantics')
            inf,bt,value,result,ft,nan=repair.groups()
            if bt!=self.bool or ft!=self.float or self.d[nan][0]!='OpConstant':
                raise RuntimeError('Unknown exceptional repair value')
            if self.d[nan][1] != [self.float, '0x1.8p+128']:
                raise RuntimeError('Expected canonical FP32 quiet NaN repair')
            hd = definitions(body)
            def one(op):
                matches = [(i,a) for i,(o,a) in hd.items() if o == op]
                if len(matches) != 1:
                    raise RuntimeError(f'Unknown helper {op} structure')
                return matches[0]
            parameter, pa = one('OpFunctionParameter')
            temporary, va = one('OpVariable')
            phi, ia = one('OpPhi')
            increment, inc = one('OpIAdd')
            length, la = one('OpCooperativeMatrixLengthKHR')
            condition, ca = one('OpULessThan')
            ret_type = hd[fn][1][0]
            if (pa != [va[0]] or va[1:] != ['Function']
                or self.d[pa[0]] != ('OpTypePointer', ['Function', ret_type])
                or self.mats[ret_type][0] != self.float
                or ia[0] != self.uint or self.value(ia[1]) != 0
                or ia[3] != increment or inc[:2] != [self.uint, phi]
                or self.value(inc[2]) != 1
                or la != [self.uint, ret_type] or ca != [self.bool, increment, length]):
                raise RuntimeError('Unknown helper induction/type data flow')
            source_op, source_args = hd[value]
            source_ptr = source_args[1]
            store = re.findall(r'OpStore (%\w+) (%\w+)', body)
            returned = re.findall(r'OpReturnValue (%\w+)', body)
            if (source_op != 'OpLoad' or source_args[0] != self.float
                or hd[source_ptr][0] != 'OpInBoundsAccessChain'
                or hd[source_ptr][1][1:] != [parameter, phi]
                or len(store) != 1 or store[0][1] != result
                or hd[store[0][0]][0] != 'OpInBoundsAccessChain'
                or hd[store[0][0]][1][1:] != [temporary, phi]
                or len(returned) != 1 or hd[returned[0]] != ('OpLoad', [ret_type, temporary])):
                raise RuntimeError('Unknown helper component data flow')
            callers=[i for i,(op,a) in self.d.items() if op=='OpFunctionCall' and a[1]==fn]
            inputs={self.d[i][1][1] for i in targets}
            if not set(callers)<=inputs:
                raise RuntimeError('Helper has uses outside saturated conversion')
            lines=[]
            is_nan=self.emit(lines,'OpIsNan',self.bool,value)
            is_inf=self.emit(lines,'OpIsInf',self.bool,value)
            special=self.emit(lines,'OpLogicalOr',self.bool,is_nan,is_inf)
            clamped=self.emit(lines,'OpExtInst',self.float,ext[0],'NClamp',value,self.constant(-448,self.float),self.constant(448,self.float))
            lines.append(f'{result} = OpSelect {self.float} {special} {nan} {clamped}')
            body=body[:repair.start()]+'\n'.join(lines)+body[repair.end():]
            self.text=self.text[:match.start()]+body+self.text[match.end():]
        for i in targets:
            self.text=re.sub(rf'^.*OpDecorate {i} SaturatedToLargestFloat8NormalConversionEXT\s*$', '',self.text,flags=re.M)
        self.report['saturated_conversions']=len(targets)

    def lengths(self):
        def replace(m):
            ty,mat=m.groups()
            if mat not in self.mats:
                raise RuntimeError('Unknown matrix length')
            return f'OpCopyObject {ty} {self.constant(8,ty)}'
        self.text,n=re.subn(r'OpCooperativeMatrixLengthKHR (%\w+) (%\w+)',replace,self.text)
        self.report['fixed_lengths']=n

    def finish(self):
        pos=re.search(r'^\s*%\w+ = OpFunction ',self.text,re.M).start()
        return self.text[:pos]+'\n'+'\n'.join(self.globals)+'\n'+self.text[pos:]

def expand_equal_loops(text):
    """Unroll bounded loops with one induction variable, starting with inner loops.

    Leave unsupported control flow to spirv-opt.
    """
    expanded=[]
    while True:
        d=definitions(text)
        const={i:int(a[1]) for i,(op,a) in d.items() if op=='OpConstant' and d.get(a[0],('',[]))[0]=='OpTypeInt'}
        for i,(op,a) in d.items():
            if op=='OpCopyObject' and a[1] in const: const[i]=const[a[1]]
        labels=list(re.finditer(r'^\s*(%\w+) = OpLabel\s*$',text,re.M))
        candidates=[]
        for n,label in enumerate(labels):
            block=text[label.end():labels[n+1].start() if n+1<len(labels) else len(text)]
            merge=re.search(r'OpLoopMerge (%\w+) (%\w+) (?:None|Unroll)',block)
            if not merge: continue
            end=re.search(rf'^\s*{merge[1]} = OpLabel\s*$',text,re.M)
            if not end or end.start()<=label.start(): continue
            body=text[label.start():end.start()]
            if body.count('OpLoopMerge')<1: continue
            phis=re.findall(r'(%\w+) = OpPhi (%\w+) (%\w+) (%\w+) (%\w+) (%\w+)',block)
            if len(phis)!=1: continue
            phi,ty,start,entry,inc,back=phis[0]
            if start not in const or d.get(inc)!=('OpIAdd',[ty,phi,next((i for i,v in const.items() if v==1 and d[i][1][0]==ty),'')]): continue
            tests=[(i,a) for i,(op,a) in d.items() if op in ['OpIEqual','OpULessThan'] and a[1]==inc and a[2] in const]
            tests=[(i,a) for i,a in tests if re.search(rf'OpBranchConditional {i} ' + (rf'{merge[1]} {label[1]}' if d[i][0]=='OpIEqual' else rf'{label[1]} {merge[1]}') + r'\b',body)]
            if len(tests)!=1: continue
            cond,a=tests[0]
            lo,hi=const[start],const[a[2]]
            if not 0<=lo<hi<=32: continue
            # Reject extra exits, returns or phi nodes at the merge.
            if len(re.findall(rf'{merge[1]}\b',body))!=2 or 'OpReturn' in body: continue
            after=text[end.end():]
            if re.match(r'\s*%\w+ = OpPhi',after): continue
            candidates.append((label,end,merge,phi,ty,cond,lo,hi,body))
        if not candidates: break
        label,end,merge,phi,ty,cond,lo,hi,body=min(candidates,key=lambda x:len(x[-1]))
        ids=re.findall(r'^\s*(%\w+) = ',body,re.M)
        prefix,suffix=text[:label.start()],text[end.start():]
        annotations=[]
        for line in prefix.splitlines():
            m=re.search(r'OpDecorate (%\w+) ',line)
            if m and m[1] in ids:
                annotations.append(line)
                prefix=prefix.replace(line+'\n','')
        copies=[]; decos=[]
        constants={v:i for i,v in const.items() if d[i][1][0]==ty}
        if any(i not in constants for i in range(lo,hi)): break
        for iteration in range(lo,hi):
            mapping={i:f'%expand_{label[1][1:]}_{iteration}_{i[1:]}' for i in ids}
            mapping[phi]=constants[iteration]
            clone=re.sub(rf'^.*{phi} = OpPhi[^\n]*\n','',body,flags=re.M)
            clone=re.sub(rf'^.*OpLoopMerge {merge[1]} %\w+ (?:None|Unroll)\s*$','',clone,flags=re.M)
            target=f'%expand_{label[1][1:]}_{iteration+1}_{label[1][1:]}' if iteration+1<hi else merge[1]
            clone=re.sub(rf'OpBranchConditional {cond} %\w+ %\w+', 'OpBranch %next_iteration',clone)
            rename=lambda s: re.sub(r'%\w+',lambda m:mapping.get(m[0],m[0]),s)
            copies.append(rename(clone).replace('%next_iteration',target))
            decos.extend(rename(x) for x in annotations)
        prefix=re.sub(rf'{label[1]}\b',f'%expand_{label[1][1:]}_{lo}_{label[1][1:]}',prefix)
        if decos:
            pos=prefix.index('OpDecorate')
            prefix=prefix[:pos]+'\n'.join(decos)+'\n'+prefix[pos:]
        suffix=re.sub(r'%\w+',lambda m:mapping.get(m[0],m[0]),suffix)
        text=prefix+'\n'.join(copies)+suffix
        expanded.append({'start':lo,'end':hi})
        if len(text)>30000000: raise RuntimeError('Expansion size limit exceeded')
    return text,expanded

def patch(source, output, expand=None, *, loads=True, stores=True, conversions=True, optimize=True,
          lds_cache=False, lds_min_groups=20, lds_min_savings=36):
    spec=importlib.util.spec_from_file_location('packed_width',Path(__file__).resolve().parents[2]/'shared/fp8/packed_width.py')
    wide=importlib.util.module_from_spec(spec); sys.modules[spec.name]=wide; spec.loader.exec_module(wide)
    output.parent.mkdir(parents=True,exist_ok=True)
    packed=output.with_suffix('.wide.spv')
    report=wide.patch_module(source,packed,'uvec2',discover_bindings=True)
    module=Module(run(['spirv-dis','--raw-id',packed]))
    if lds_cache:
        module.lds_repeated_matrix_cache(lds_min_groups, lds_min_savings)
    else:
        module.report['lds_cache']={'matched':False,'reason':'disabled'}
    module.memory(loads, stores)
    if conversions: module.conversions()
    module.lengths()
    if expand is None:
        # Leave FP8 accumulator loops for later optimizer passes.
        expand = not any(
            module.mats[module.d[a[1]][1][0]][0] == module.fp8
            and module.value(module.mats[module.d[a[1]][1][0]][4]) == 2
            for a in (m.split() for m in re.findall(r'OpCooperativeMatrixStoreKHR ([^\n]+)', module.text))
        )
    text=module.finish()
    if expand: text,loops=expand_equal_loops(text)
    else: loops=[]
    text=re.sub(r'(OpLoopMerge %\w+ %\w+) None',r'\1 Unroll',text)
    raw=output.with_suffix('.before.spv')
    asm=output.with_suffix('.before.spvasm'); asm.write_text(text)
    run(['spirv-as','--target-env','spv1.6',asm,'-o',raw]); run(['spirv-val',raw])
    flags=['--simplify-instructions','--loop-unroll','--eliminate-dead-code-aggressive','--simplify-instructions']
    if optimize: flags += ['-O']
    candidate = output.with_suffix('.candidate.spv')
    run(['spirv-opt',*flags,raw,'-o',candidate])
    run(['spirv-val',candidate])
    candidate.replace(output)
    output.with_suffix('.spvasm').write_text(run(['spirv-dis','--raw-id',output]))
    report.update(module.report); report['expanded_loops']=loops
    report['manual_expansion_enabled'] = expand
    report['output'] = str(output)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--no-expand',action='store_true')
    p.add_argument('--lds-cache',action='store_true',help='cache repeated WorkgroupId-dependent FP8 matrix loads in LDS')
    p.add_argument('--lds-min-groups',type=int,default=20)
    p.add_argument('--lds-min-savings',type=int,default=36)
    a=p.parse_args(); print(json.dumps(patch(
        a.input,a.output,False if a.no_expand else None,
        lds_cache=a.lds_cache,lds_min_groups=a.lds_min_groups,lds_min_savings=a.lds_min_savings),indent=2))

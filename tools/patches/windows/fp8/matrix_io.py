#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Optimize FP8 matrix loads for the tested AMD Windows Wave32 layout.

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
    def __init__(self, text, *, require_local_invocation=True, preserve_names=False):
        if not preserve_names:
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
        if not require_local_invocation:
            self.lid = None
            self.lidtype = None
        elif len(self.lid) != 1:
            raise RuntimeError('Requires one LocalInvocationId')
        else:
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

    def _value_type(self, ident):
        op,a=self.d.get(ident,('',[]))
        if op=='OpVariable':
            if not a: raise RuntimeError(f'variable has no pointer type: {ident}')
            return a[0]
        if a and a[0].startswith('%'):
            return a[0]
        raise RuntimeError(f'cannot determine value type for {ident}: {op} {a}')

    def _decoration_value(self, target, decoration):
        vals=re.findall(rf'^\s*OpDecorate {re.escape(target)} {re.escape(decoration)} (\d+)\s*$',self.text,re.M)
        if not vals: return None
        if len(set(vals))!=1: raise RuntimeError(f'conflicting {decoration} decorations for {target}: {vals}')
        return int(vals[0])

    def _member_offset(self, struct_ty, member):
        vals=re.findall(rf'^\s*OpMemberDecorate {re.escape(struct_ty)} {member} Offset (\d+)\s*$',self.text,re.M)
        if not vals: return None
        if len(set(vals))!=1: raise RuntimeError(f'conflicting member offsets for {struct_ty}[{member}]: {vals}')
        return int(vals[0])

    def _has_decoration(self, target, decoration):
        return re.search(rf'^\s*OpDecorate {re.escape(target)} {re.escape(decoration)}(?:\s+.*)?$',self.text,re.M) is not None

    def _type_debug(self, ident):
        op,args=self.d.get(ident,('',[]))
        decos=[line.strip() for line in self.text.splitlines()
               if re.match(rf'^\s*Op(?:Member)?Decorate {re.escape(ident)}\b',line)]
        return {'id':ident,'op':op,'args':args,'decorations':decos}

    def accumulator_address(self, lines, ptr, stride):
        """Rewrite packed FP8 accumulator accesses for Wave32.

        Descriptor arrays need no byte stride. Require 8-byte buffer alignment
        and a uvec2 payload with stride 8. Only the final payload index gains
        the row and half offsets.
        """
        op,a=self.d.get(ptr,('',[]))
        if op not in ['OpAccessChain','OpInBoundsAccessChain'] or len(a)<3:
            raise RuntimeError(f'unsupported accumulator pointer definition: {ptr} = {op} {a}')
        result_ptr_ty,base,*indices=a
        rtop,rargs=self.d.get(result_ptr_ty,('',[]))
        if rtop!='OpTypePointer' or rargs!=['StorageBuffer',self.vec]:
            raise RuntimeError(f'accumulator result pointer is not StorageBuffer uvec2: {result_ptr_ty} -> {rtop} {rargs}')
        if not indices: raise RuntimeError('accumulator access chain has no indices')
        roots=self._root_variables(base)
        if len(roots)!=1:
            raise RuntimeError(f'accumulator access chain must have one root variable, got {sorted(roots)}')
        root=next(iter(roots))
        rop,rva=self.d.get(root,('',[]))
        if rop!='OpVariable' or len(rva)<2 or rva[1]!='StorageBuffer':
            raise RuntimeError(f'accumulator root is not StorageBuffer variable: {root} -> {rop} {rva}')
        descriptor_set=self._decoration_value(root,'DescriptorSet')
        binding=self._decoration_value(root,'Binding')
        if descriptor_set is None or binding is None:
            raise RuntimeError(f'accumulator root lacks DescriptorSet/Binding: {root}')
        base_ty=self._value_type(base)
        btop,bargs=self.d.get(base_ty,('',[]))
        if btop!='OpTypePointer' or len(bargs)!=2 or bargs[0]!='StorageBuffer':
            raise RuntimeError(f'accumulator access base is not StorageBuffer pointer: {base_ty} -> {btop} {bargs}')
        current=bargs[1]
        proof=[]; descriptor_array_hops=0
        for depth,idx in enumerate(indices[:-1]):
            top,args=self.d.get(current,('',[]))
            if top=='OpTypeStruct':
                member=self.value(idx)
                if member is None: raise RuntimeError(f'struct access index is not constant: {idx}')
                if member<0 or member>=len(args): raise RuntimeError(f'struct member {member} out of range for {current}')
                off=self._member_offset(current,member)
                if off is None: raise RuntimeError(f'missing Offset decoration for {current}[{member}]')
                if off%8: raise RuntimeError(f'unaligned struct member offset {off} for {current}[{member}]')
                proof.append({'depth':depth,'kind':'struct','type':current,'index':idx,'member':member,'offset':off})
                current=args[member]; continue
            if top in ['OpTypeRuntimeArray','OpTypeArray']:
                if not args: raise RuntimeError(f'malformed aggregate type {current}: {top} {args}')
                elem=args[0]
                arr_stride=self._decoration_value(current,'ArrayStride')
                if arr_stride is None:
                    etop,_=self.d.get(elem,('',[]))
                    if descriptor_array_hops!=0 or etop!='OpTypeStruct' or not self._has_decoration(elem,'Block'):
                        raise RuntimeError(f'array without ArrayStride is not proven outer descriptor array: {self._type_debug(current)}')
                    proof.append({'depth':depth,'kind':'descriptor-array','type':current,'index':idx,
                                  'element_block_type':elem,'descriptor_set':descriptor_set,'binding':binding})
                    descriptor_array_hops+=1; current=elem; continue
                if arr_stride%8: raise RuntimeError(f'unaligned aggregate stride {arr_stride} for {current}')
                proof.append({'depth':depth,'kind':top,'type':current,'index':idx,'array_stride':arr_stride})
                current=elem; continue
            raise RuntimeError(f'unsupported aggregate before accumulator payload at depth {depth}: {self._type_debug(current)}')
        leaf_top,leaf_args=self.d.get(current,('',[]))
        if leaf_top not in ['OpTypeRuntimeArray','OpTypeArray'] or not leaf_args or leaf_args[0]!=self.vec:
            raise RuntimeError(f'final accumulator index does not select array<uvec2>: {self._type_debug(current)}')
        leaf_stride=self._decoration_value(current,'ArrayStride')
        if leaf_stride!=8:
            raise RuntimeError(f'accumulator payload ArrayStride is {leaf_stride}, expected 8')
        lid=self.emit(lines,'OpLoad',self.lidtype,self.lid)
        lane=self.emit(lines,'OpCompositeExtract',self.uint,lid,0)
        row=self.emit(lines,'OpBitwiseAnd',self.uint,lane,self.constant(15))
        half=self.emit(lines,'OpShiftRightLogical',self.uint,lane,self.constant(4))
        offset=self.emit(lines,'OpIMul',self.uint,row,self.constant(stride))
        offset=self.emit(lines,'OpIAdd',self.uint,offset,half)
        index=self.emit(lines,'OpIAdd',self.uint,indices[-1],offset)
        address=self.emit(lines,op,result_ptr_ty,base,*indices[:-1],index)
        return address,{'access_op':op,'root_variable':root,'descriptor_set':descriptor_set,'binding':binding,
                        'original_indices':indices,'leaf_container_type':current,'leaf_array_stride':leaf_stride,
                        'descriptor_array_hops':descriptor_array_hops,'preserved_path_proof':proof}

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

    def _guaranteed_tz(self, ident, seen=None):
        """Count the low bits known to be zero in an unsigned integer."""
        seen=set() if seen is None else seen
        if ident in seen:
            return 0
        seen.add(ident)
        value=self.value(ident)
        if value is not None:
            value &= 0xffffffff
            if value==0:
                return 32
            return min(32,(value & -value).bit_length()-1)
        op,a=self.d.get(ident,('',[]))
        if op in ['OpCopyObject','OpBitcast'] and len(a)>=2:
            return self._guaranteed_tz(a[1],seen)
        if op in ['OpIAdd','OpISub','OpBitwiseOr','OpBitwiseXor'] and len(a)>=3:
            return min(self._guaranteed_tz(a[1],seen.copy()),
                       self._guaranteed_tz(a[2],seen.copy()))
        if op=='OpIMul' and len(a)>=3:
            return min(32,self._guaranteed_tz(a[1],seen.copy())+
                          self._guaranteed_tz(a[2],seen.copy()))
        if op=='OpBitwiseAnd' and len(a)>=3:
            return max(self._guaranteed_tz(a[1],seen.copy()),
                       self._guaranteed_tz(a[2],seen.copy()))
        if op=='OpShiftLeftLogical' and len(a)>=3:
            shift=self.value(a[2])
            if shift is not None:
                return min(32,self._guaranteed_tz(a[1],seen.copy())+shift)
        if op in ['OpShiftRightLogical','OpShiftRightArithmetic'] and len(a)>=3:
            shift=self.value(a[2])
            if shift is not None:
                return max(0,self._guaranteed_tz(a[1],seen.copy())-shift)
        if op=='OpSelect' and len(a)>=4:
            return min(self._guaranteed_tz(a[2],seen.copy()),
                       self._guaranteed_tz(a[3],seen.copy()))
        return 0

    def _affine_form(self, ident, seen=None):
        """Represent an address as symbolic terms and a constant.

        Track exact integer operations; keep unsupported expressions as opaque terms.
        """
        seen=set() if seen is None else seen
        if ident in seen:
            return {('rec',ident):1},0
        seen.add(ident)

        value=self.value(ident)
        if value is not None:
            return {},value

        op,a=self.d.get(ident,('',[]))
        if op in ['OpCopyObject','OpBitcast'] and len(a)>=2:
            return self._affine_form(a[1],seen)

        if op in ['OpIAdd','OpISub'] and len(a)>=3:
            lt,lc=self._affine_form(a[1],seen.copy())
            rt,rc=self._affine_form(a[2],seen.copy())
            sign=1 if op=='OpIAdd' else -1
            terms=dict(lt)
            for term,coeff in rt.items():
                terms[term]=terms.get(term,0)+sign*coeff
                if not terms[term]: del terms[term]
            return terms,lc+sign*rc

        if op=='OpBitwiseOr' and len(a)>=3:
            lv=self.value(a[1]); rv=self.value(a[2])
            if lv is not None and rv is None:
                # OR equals addition when the dynamic operand has no bits set in the constant.
                needed=lv.bit_length()
                if self._guaranteed_tz(a[2])>=needed:
                    terms,const=self._affine_form(a[2],seen.copy())
                    return terms,const+lv
            if rv is not None and lv is None:
                needed=rv.bit_length()
                if self._guaranteed_tz(a[1])>=needed:
                    terms,const=self._affine_form(a[1],seen.copy())
                    return terms,const+rv

        if op=='OpIMul' and len(a)>=3:
            lt,lc=self._affine_form(a[1],seen.copy())
            rt,rc=self._affine_form(a[2],seen.copy())
            if not lt:
                scale=lc
                return {k:v*scale for k,v in rt.items()},rc*scale
            if not rt:
                scale=rc
                return {k:v*scale for k,v in lt.items()},lc*scale

        if op=='OpShiftLeftLogical' and len(a)>=3:
            shift=self.value(a[2])
            if shift is not None and 0<=shift<32:
                terms,const=self._affine_form(a[1],seen.copy())
                scale=1<<shift
                return {k:v*scale for k,v in terms.items()},const*scale

        if op in ['OpShiftRightLogical','OpShiftRightArithmetic'] and len(a)>=3:
            shift=self.value(a[2])
            if shift is not None and 0<=shift<32:
                terms,const=self._affine_form(a[1],seen.copy())
                scale=1<<shift
                # A right shift is exact only when every coefficient and constant is divisible.
                if const%scale==0 and all(v%scale==0 for v in terms.values()):
                    return {k:v//scale for k,v in terms.items()},const//scale

        # Keep unknown expressions as opaque terms so shared terms still cancel.
        return {self._canonical(ident):1},0

    @staticmethod
    def _term_key(terms):
        return tuple(sorted(((repr(k),v) for k,v in terms.items())))

    @staticmethod
    def _find_strided_rectangle(constants, counts, matrix_span, step,
                                min_groups, min_savings):
        """Find a complete rectangular group of matrix addresses.

        Infer the row pitch and matrix starts shared by consecutive rows.
        """
        values=sorted(set(constants))
        if len(values)<min_groups:
            return None

        # Repeated address differences suggest a row pitch; validate the full rectangle below.
        from collections import Counter
        diffs=Counter()
        for i,a in enumerate(values):
            for b in values[i+1:]:
                d=b-a
                if d>matrix_span:
                    diffs[d]+=1
        if not diffs:
            return None

        best=None
        for pitch,_freq in diffs.most_common():
            if pitch<=matrix_span:
                continue
            origin=min(values)
            rows={}
            for value in values:
                q,r=divmod(value-origin,pitch)
                rows.setdefault(q,set()).add(r)
            row_ids=sorted(rows)
            if len(row_ids)<3:
                continue

            # Find matrix starts shared by every row in each candidate range.
            for ai in range(len(row_ids)):
                if ai and row_ids[ai]!=row_ids[ai-1]+1:
                    pass
                common=None
                previous=None
                for bi in range(ai,len(row_ids)):
                    q=row_ids[bi]
                    if previous is not None and q!=previous+1:
                        break
                    previous=q
                    common=set(rows[q]) if common is None else common & rows[q]
                    if not common or len(common)<2:
                        break
                    row_count=bi-ai+1
                    if row_count<3:
                        continue
                    cols=sorted(common)
                    # Require matrix starts to share the packed element alignment.
                    c0=cols[0]
                    if any((c-c0)%step for c in cols):
                        continue
                    width=cols[-1]-c0+matrix_span
                    if width<=0 or width%2 or width//2>32:
                        continue
                    starts={origin+q*pitch+c for q in row_ids[ai:bi+1] for c in cols}
                    unique=len(starts)
                    redirected=sum(counts.get(x,0) for x in starts)
                    savings=redirected-unique
                    if unique<min_groups or savings<min_savings:
                        continue
                    score=(redirected,unique,savings,row_count,len(cols),-width,-pitch)
                    candidate={
                        'score':score,
                        'pitch':pitch,
                        'origin':origin,
                        'row_first':row_ids[ai],
                        'rows':row_count,
                        'cols':cols,
                        'col_first':c0,
                        'row_width':width,
                        'starts':starts,
                        'unique':unique,
                        'redirected':redirected,
                        'savings':savings,
                    }
                    if best is None or candidate['score']>best['score']:
                        best=candidate
        return best

    def lds_repeated_matrix_cache(self, min_groups=20, min_savings=36):
        """Copy a rectangular Matrix B window into LDS.

        Based on Yuseng's optimization. Each participating lane copies 16 bytes.
        Synchronize once before redirecting matrix loads to the shared copy.
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
            # Rewrite only Matrix B activations; leave weights and accumulators unchanged.
            if role!=1 or self.value(layout)!=role:
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
            op,pa=self.d.get(ptr,('',[]))
            if op not in ['OpAccessChain','OpInBoundsAccessChain'] or len(pa)!=4:
                continue
            if self.value(pa[2])!=0:
                continue
            step=self.value(stride)
            if step is None or step<2:
                continue
            terms,const=self._affine_form(pa[3])
            if not terms:
                continue
            records.append({
                'line':n,'block':block,'result':result,'type':ty,'ptr':ptr,
                'layout':layout,'stride':stride,'step':step,'flags':flags,
                'role':role,'root':roots[0],'ptr_op':op,'ptr_type':pa[0],
                'buffer_base':pa[1],'zero_index':pa[2],'index_id':pa[3],
                'terms':terms,'term_key':self._term_key(terms),'const':const,
            })

        clusters={}
        for r in records:
            key=(r['block'],r['root'],r['role'],r['step'],r['term_key'])
            clusters.setdefault(key,[]).append(r)

        viable=[]
        for key,items in clusters.items():
            counts=Counter(r['const'] for r in items)
            step=items[0]['step']
            # Each of the 16 rows spans two uvec2 elements, so the last row ends at 15 * stride + 2.
            matrix_span=15*step+2
            rect=self._find_strided_rectangle(
                list(counts),counts,matrix_span,step,min_groups,min_savings)
            if rect:
                viable.append((rect['score'],key,items,rect))

        if not viable:
            self.report['lds_cache']={
                'matched':False,
                'reason':'no complete affine 2D Matrix-B window',
                'min_groups':min_groups,'min_savings':min_savings,
            }
            return
        viable.sort(reverse=True,key=lambda x:x[0])
        if len(viable)>1 and viable[0][0]==viable[1][0]:
            raise RuntimeError('Ambiguous 2D strided LDS candidate')

        _score,(target_block,root,role,step,_term_key),items,rect=viable[0]
        origin=rect['origin']
        first_row=rect['row_first']
        pitch=rect['pitch']
        col0=rect['col_first']
        rows=rect['rows']
        cols=rect['cols']
        row_width=rect['row_width']
        grid_base_const=origin+first_row*pitch+col0
        starts=rect['starts']

        targets=[r for r in items if r['const'] in starts]
        target_results={r['result']:r for r in targets}
        first=min(r['line'] for r in targets)
        last=max(r['line'] for r in targets)

        # Every load in the tile must use the same descriptor access-chain shape.
        exemplar=min((r for r in targets if r['const']==grid_base_const),
                     key=lambda r:r['line'],default=None)
        if exemplar is None:
            raise RuntimeError('Cannot locate 2D LDS grid origin')
        shape=(exemplar['ptr_op'],exemplar['ptr_type'],exemplar['buffer_base'],exemplar['zero_index'])
        if any((r['ptr_op'],r['ptr_type'],r['buffer_base'],r['zero_index'])!=shape for r in targets):
            raise RuntimeError('2D LDS grid uses incompatible source pointer chains')

        # Do not reuse a source buffer written before the last matrix load.
        for line in lines[:last+1]:
            sm=re.match(r'\s*OpStore (%\w+) ',line)
            cm=re.match(r'\s*OpCooperativeMatrixStoreKHR (%\w+) ',line)
            ptr=(sm or cm)
            if ptr and root in self._root_variables(ptr[1]):
                raise RuntimeError('2D LDS source is written before its last redirected load')

        total_elements=rows*row_width
        count_id=self.constant(total_elements)
        array_ty=self.fresh()
        self.d[array_ty]=('OpTypeArray',[self.vec,count_id])
        self.globals.append(f'{array_ty} = OpTypeArray {self.vec} {count_id}')
        array_ptr=self.fresh()
        self.d[array_ptr]=('OpTypePointer',['Workgroup',array_ty])
        self.globals.append(f'{array_ptr} = OpTypePointer Workgroup {array_ty}')
        elem_ptrs=[i for i,(op,a) in self.d.items()
                   if op=='OpTypePointer' and a==['Workgroup',self.vec]]
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

        # SPIR-V 1.4 and later require Workgroup globals in the entry-point interface.
        entry_points=[]
        for n,line in enumerate(lines):
            tokens=line.strip().split()
            if len(tokens)>=4 and tokens[0]=='OpEntryPoint' and tokens[1]=='GLCompute':
                entry_points.append((n,line))
        if len(entry_points)!=1:
            raise RuntimeError(
                f'2D LDS requires exactly one GLCompute entry point, found {len(entry_points)}')
        entry_index,entry_line=entry_points[0]
        lines[entry_index]=entry_line+f' {lds}'

        prefix=[]
        memo={}
        # Reuse the top-left matrix pointer so dynamic address terms stay unchanged.
        grid_index=self._clone_value(exemplar['index_id'],prefix,memo)
        if exemplar['const']!=grid_base_const:
            raise RuntimeError('Unexpected 2D LDS origin constant mismatch')

        lid=self.emit(prefix,'OpLoad',self.lidtype,self.lid)
        lane=self.emit(prefix,'OpCompositeExtract',self.uint,lid,0)
        active_lanes=row_width//2
        cond=self.emit(prefix,'OpULessThan',self.bool,lane,self.constant(active_lanes))
        then_label=self.fresh()
        merge_label=self.fresh()
        prefix.append(f'OpSelectionMerge {merge_label} None')
        prefix.append(f'OpBranchConditional {cond} {then_label} {merge_label}')
        prefix.append(f'{then_label} = OpLabel')

        lane2=self.emit(prefix,'OpShiftLeftLogical',self.uint,lane,self.constant(1))
        one=self.constant(1)
        for row in range(rows):
            src_index=grid_index
            row_offset=row*pitch
            if row_offset:
                src_index=self.emit(prefix,'OpIAdd',self.uint,src_index,self.constant(row_offset))
            src_index=self.emit(prefix,'OpIAdd',self.uint,src_index,lane2)
            src0=self.emit(prefix,exemplar['ptr_op'],exemplar['ptr_type'],
                           exemplar['buffer_base'],exemplar['zero_index'],src_index)
            packed0=self.emit(prefix,'OpLoad',self.vec,src0,'Aligned',8)
            src_index1=self.emit(prefix,'OpIAdd',self.uint,src_index,one)
            src1=self.emit(prefix,exemplar['ptr_op'],exemplar['ptr_type'],
                           exemplar['buffer_base'],exemplar['zero_index'],src_index1)
            packed1=self.emit(prefix,'OpLoad',self.vec,src1,'Aligned',8)

            dst_index=lane2
            row_tight=row*row_width
            if row_tight:
                dst_index=self.emit(prefix,'OpIAdd',self.uint,dst_index,self.constant(row_tight))
            dst0=self.emit(prefix,'OpAccessChain',elem_ptr,lds,dst_index)
            prefix.append(f'OpStore {dst0} {packed0} Aligned 8')
            dst_index1=self.emit(prefix,'OpIAdd',self.uint,dst_index,one)
            dst1=self.emit(prefix,'OpAccessChain',elem_ptr,lds,dst_index1)
            prefix.append(f'OpStore {dst1} {packed1} Aligned 8')

        prefix.append(f'OpBranch {merge_label}')
        prefix.append(f'{merge_label} = OpLabel')
        # Synchronize all 32 lanes before reading the shared tile.
        prefix.append(f'OpControlBarrier {self.constant(2)} {self.constant(2)} {self.constant(264)}')

        # Update downstream OpPhi predecessors after splitting this block.
        for n,line in enumerate(lines):
            if ' OpPhi ' in line:
                tokens=line.split()
                for i,tok in enumerate(tokens):
                    if tok==target_block:
                        tokens[i]=merge_label
                lines[n]=' '.join(tokens)

        out=[]
        inserted=False
        redirected=0
        for n,line in enumerate(lines):
            if n==first:
                out.extend(prefix)
                inserted=True
            mt=re.match(r'\s*(%\w+) = OpCooperativeMatrixLoadKHR (.*)',line)
            if mt and mt[1] in target_results:
                r=target_results[mt[1]]
                rel=r['const']-grid_base_const
                row_index=rel//pitch
                col_index=rel%pitch
                if not (0<=row_index<rows and col_index in cols):
                    raise RuntimeError('2D LDS target escaped proven rectangle')
                lds_index=row_index*row_width+(col_index-col0)
                lptr=self.emit(out,'OpAccessChain',elem_ptr,lds,self.constant(lds_index))
                out.append(
                    f"{r['result']} = OpCooperativeMatrixLoadKHR {r['type']} "
                    f"{lptr} {r['layout']} {r['stride']} {' '.join(r['flags'])}")
                redirected+=1
            else:
                out.append(line)
        if not inserted or redirected!=len(target_results):
            raise RuntimeError('Failed to rewrite complete 2D LDS target set')

        self.text='\n'.join(out)+'\n'
        self.report['lds_cache']={
            'matched':True,
            'mode':'strided-2d-uvec2-pairs',
            'block':target_block,
            'matrix_role':role,
            'rows':rows,
            'matrix_columns':len(cols),
            'matrix_column_offsets':cols,
            'global_row_pitch_packed':pitch,
            'lds_row_width_packed':row_width,
            'active_loader_lanes':active_lanes,
            'staging_load_width_bytes':16,
            'staging_uses_two_adjacent_uvec2_loads':True,
            'distinct_staged_matrices':rect['unique'],
            'redirected_matrix_loads':redirected,
            'global_matrix_loads_saved':rect['savings'],
            'lds_bytes':total_elements*8,
            'source_root':root,
            'min_groups':min_groups,
            'min_savings':min_savings,
        }

    def memory(self, loads=True, stores=True, packed_a_words=False, *,
               accumulator_io=False, components_first=True):
        count={'loads':0,'stores':0,'retained_loads':0,'retained_stores':0,
               'packed_a_word_loads':0,'accumulator_loads':0,'accumulator_stores':0}
        accumulator_proofs=[]
        fp8vec4 = None
        if packed_a_words:
            vectors = [i for i,(op,a) in self.d.items()
                       if op == 'OpTypeVector' and a == [self.fp8, '4']]
            if len(vectors) > 1:
                raise RuntimeError('Ambiguous vec4<FP8> type')
            if vectors:
                fp8vec4 = vectors[0]
            else:
                fp8vec4 = self.fresh()
                self.d[fp8vec4] = ('OpTypeVector',[self.fp8,'4'])
                self.globals.append(f'{fp8vec4} = OpTypeVector {self.fp8} 4')
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
            matrix_ab=(role in [0,1] and self.value(layout)==role)
            accumulator=(accumulator_io and role==2 and self.value(layout)==1)
            eligible=(mat[0]==self.fp8 and (matrix_ab or accumulator)
                      and storage=='StorageBuffer' and (loads if mt else stores))
            if not eligible:
                count['retained_loads' if mt else 'retained_stores']+=1
                out.append(line); continue
            step=self.value(stride)
            if step is None or step < 2 or flags != ['NonPrivatePointer']:
                raise RuntimeError('Unsupported stride or memory operands')
            lines=[]
            if role==2:
                if step!=4:
                    raise RuntimeError(f'Unsupported packed accumulator stride {step}; expected 4')
                address,proof=self.accumulator_address(lines,ptr,step)
                proof['kind']='load' if mt else 'store'
                accumulator_proofs.append(proof)
            else:
                address=self.address(lines,ptr,step)
            if mt:
                packed=self.emit(lines,'OpLoad',self.vec,address,'Aligned',8)
                words=[self.emit(lines,'OpCompositeExtract',self.uint,packed,w) for w in range(2)]
                previous=self.undef(ty)
                values=[]
                if packed_a_words and role == 0:
                    # Keep four FP8 values packed in each 32-bit word, as in Yuseng's Matrix A optimization.
                    for word in words:
                        packed_fp8=self.emit(lines,'OpBitcast',fp8vec4,word)
                        values.extend(
                            self.emit(lines,'OpCompositeExtract',self.fp8,packed_fp8,b)
                            for b in range(4))
                    count['packed_a_word_loads']+=1
                else:
                    for i in range(8):
                        word=words[i//4]
                        if i%4:
                            word=self.emit(lines,'OpShiftRightLogical',self.uint,word,self.constant(i%4*8))
                        byte=self.emit(lines,'OpUConvert',self.byte,word)
                        value=self.emit(lines,'OpBitcast',self.fp8,byte)
                        if components_first:
                            values.append(value)
                        else:
                            result=mt[1] if i==7 else self.fresh()
                            lines.append(f'{result} = OpCompositeInsert {ty} {value} {previous} {i}')
                            previous=result
                for i,value in enumerate(values):
                    result=mt[1] if i==7 else self.fresh()
                    lines.append(f'{result} = OpCompositeInsert {ty} {value} {previous} {i}')
                    previous=result
                count['loads']+=1
                if role==2: count['accumulator_loads']+=1
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
                if role==2: count['accumulator_stores']+=1
            out.extend(lines)
        self.text='\n'.join(out)+'\n'
        self.report['memory']=count
        self.report['accumulator_address_proofs']=accumulator_proofs

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

def expand_equal_loops(text, *, helpers_only=False, max_iterations=32):
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
            if hi-lo > max_iterations: continue
            if helpers_only:
                functions=list(re.finditer(r'(%\w+) = OpFunction (%\w+) ',text[:label.start()]))
                if not functions or d.get(functions[-1][2], ('',[]))[0]!='OpTypeCooperativeMatrixKHR':
                    continue
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
          lds_cache=False, lds_min_groups=20, lds_min_savings=36,
          loop_control='unroll', loop_pass=True, accumulator_io=False,
          components_first=True, preserve_names=False, packed_a_words=False):
    spec=importlib.util.spec_from_file_location('packed_width',Path(__file__).resolve().parents[2]/'shared/fp8/packed_width.py')
    wide=importlib.util.module_from_spec(spec); sys.modules[spec.name]=wide; spec.loader.exec_module(wide)
    output.parent.mkdir(parents=True,exist_ok=True)
    packed=output.with_suffix('.wide.spv')
    report=wide.patch_module(source,packed,'uvec2',discover_bindings=True)
    module=Module(run(['spirv-dis','--raw-id',packed]), preserve_names=preserve_names)
    if lds_cache:
        module.lds_repeated_matrix_cache(lds_min_groups, lds_min_savings)
    else:
        module.report['lds_cache']={'matched':False,'reason':'disabled'}
    module.memory(loads, stores, packed_a_words=packed_a_words or lds_cache,
                  accumulator_io=accumulator_io, components_first=components_first)
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
    if loop_control not in ('unroll', 'none'):
        raise RuntimeError('loop_control must be unroll or none')
    if loop_control == 'unroll':
        text=re.sub(r'(OpLoopMerge %\w+ %\w+) None',r'\1 Unroll',text)
    else:
        text=re.sub(r'(OpLoopMerge %\w+ %\w+) Unroll',r'\1 None',text)
    raw=output.with_suffix('.before.spv')
    asm=output.with_suffix('.before.spvasm'); asm.write_text(text)
    run(['spirv-as','--target-env','spv1.6',asm,'-o',raw]); run(['spirv-val',raw])
    flags=['--simplify-instructions']
    if loop_pass: flags += ['--loop-unroll']
    flags += ['--eliminate-dead-code-aggressive','--simplify-instructions']
    if optimize: flags += ['-O']
    candidate = output.with_suffix('.candidate.spv')
    run(['spirv-opt',*flags,raw,'-o',candidate])
    run(['spirv-val',candidate])
    candidate.replace(output)
    output.with_suffix('.spvasm').write_text(run(['spirv-dis','--raw-id',output]))
    report.update(module.report); report['expanded_loops']=loops
    report['manual_expansion_enabled'] = expand
    report['loop_control'] = loop_control
    report['loop_pass'] = loop_pass
    report['accumulator_io'] = accumulator_io
    report['components_first'] = components_first
    report['preserve_names'] = preserve_names
    report['final_O'] = optimize
    report['output'] = str(output)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--no-expand',action='store_true')
    p.add_argument('--loop-control',choices=['unroll','none'],default='unroll')
    p.add_argument('--no-loop-pass',action='store_true')
    p.add_argument('--no-optimize',action='store_true')
    p.add_argument('--accumulator-io',action='store_true')
    p.add_argument('--packed-a-words',action='store_true')
    p.add_argument('--lds-cache',action='store_true',help='stage proven 2D strided FP8 Matrix-B windows in compact LDS')
    p.add_argument('--lds-min-groups',type=int,default=20)
    p.add_argument('--lds-min-savings',type=int,default=36)
    a=p.parse_args(); print(json.dumps(patch(
        a.input,a.output,False if a.no_expand else None,
        lds_cache=a.lds_cache,lds_min_groups=a.lds_min_groups,lds_min_savings=a.lds_min_savings,
        loop_control=a.loop_control,loop_pass=not a.no_loop_pass,optimize=not a.no_optimize,
        accumulator_io=a.accumulator_io,packed_a_words=a.packed_a_words),indent=2))

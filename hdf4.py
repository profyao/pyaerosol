'''
This module provides READONLY access to HDF4 data file content using
the pyhdf package.  The intent is to correct or overcome certain
deficiencies in the pyhdf support.  These deficiencies include:

* provide a view of the structure of the data tree
* provide consistent nomenclature for different structure types

The HDF4 support library is inconsistent amongst the different
data structure types.  This module declares several classes,
each sharing a common nomenclature.

Example
-------

::

  import hdf4
  hdf = hdf4.HDF4_root('myhdf4file.hdf')
  print hdf          # shows the structure tree

  # show the attributes:
  for key, value in hdf.attr.items():
      print key, '=', value
'''

import os
import pyhdf.HDF
import pyhdf.SD
import pyhdf.V


SDC_TYPE = {
  pyhdf.SD.SDC.CHAR    : "8-bit character",
  pyhdf.SD.SDC.CHAR8   : "8-bit character",
  pyhdf.SD.SDC.UCHAR   : "unsigned 8-bit integer",
  pyhdf.SD.SDC.UCHAR8  : "unsigned 8-bit integer",
  pyhdf.SD.SDC.INT8    : "signed 8-bit integer",
  pyhdf.SD.SDC.UINT8   : "unsigned 8-bit integer",
  pyhdf.SD.SDC.INT16   : "signed 16-bit integer",
  pyhdf.SD.SDC.UINT16  : "unsigned 16-bit integer",
  pyhdf.SD.SDC.INT32   : "signed 32-bit integer",
  pyhdf.SD.SDC.UINT32  : "unsigned 32-bit integer",
  pyhdf.SD.SDC.FLOAT32 : "32-bit floating point",
  pyhdf.SD.SDC.FLOAT64 : "64-bit floating point",
}

TAG_TYPE = {
    pyhdf.HDF.HC.DFTAG_NDG: 'dataset',
    pyhdf.HDF.HC.DFTAG_VH: 'vdata',
    pyhdf.HDF.HC.DFTAG_VG: 'vgroup',
}

SPECIAL_CLASSES = (         # classes created internally by the HDF support library
                  'CDF0.0',         # root vgroup of the file
                  'Var0.0',
                  'Dim0.0',
                  'RIG0.0',
                  )

SPECIAL_ATTRIBUTES = (         # attributes created internally by the HDF support library
                  '_FillValue',
                  )

STRUCTURE_TYPE = {
  'root'        : 'root',
  'vgroup'      : 'vgroup',
  'SDS'         : 'SDS',
  'attribute'   : 'attribute',
}


class HDF4_Master:
    '''
    superclass of the various classes defined in this module

    do not call this class directly
    '''
    attr = {}           #: dictionary of attributes: key: object
    root_obj = None     #: root object
    hdftype = None      #: typecode of this object
    name = ""           #: HDF text name of this object (file name for root object)
    obj = None          #: pyhdf object
    path = None         #: HDF path to this object in the file
    refnum = None       #: pyhdf object reference number
    structure = None    #: one of STRUCTURE_TYPE

    def __init__(self):
        self.attr = {}

    def __str__(self):
        s = [self.path]
        for attr in sorted(self.attr.values()):
            s.append( str(attr) )
        return "\n".join(s)


class HDF4_root(HDF4_Master):
    '''
    Root element of HDF4 file

    simple usage::

        import hdf4
        hdf_document = hdf4.HDF4_root('hdf_file.hdf')

    The best algorithm to reconstruct the data tree (so far) proceeds as follows:

    1. open both V and SD interfaces to the file
    2. walk the list of vgroup nodes  (as implemented in the parse() method)
      A. if node is not in the cache
        a. ignore it if it is a ''special class'' (internal to HDF support library)
        b. otherwise create an object
        c. add the object to the cache
      B. get the object from the cache
      C. walk through the child nodes
        a. for vgroup nodes, recurse from step 2
        b. for SDS nodes, save the SDS object and attributes
    3. gather any global attribute nodes at the root level
    '''
    children = {}   #: dictionary of children: key: object
    nexus = False   #: determination that this file is NeXus-compliant

    def __init__(self, filename):
        if not os.path.exists(filename):
            raise IOError, "File not found: " + filename
        if not pyhdf.HDF.ishdf(filename):
            raise pyhdf.HDF.HDF4Error, "File not HDF4 format: " + filename

        HDF4_Master.__init__(self)

        self.name = filename
        self.root_obj = self
        self.refnum = 0
        self.path = "/"
        self.children = {}
        self.structure = STRUCTURE_TYPE["root"]

        self.hdf = pyhdf.HDF.HDF(filename)
        self.sd  = pyhdf.SD.SD(filename)
        self.v = self.hdf.vgstart()

        self.parse()        # identify all the vgroups in the file

    def __del__(self):
        self.v.end()
        self.sd.end()
        self.hdf.close()

    def __str__(self):
        s = [self.path]
        for attr in sorted(self.attr.values()):
            s.append( str(attr) )
        for child in sorted(self.children.values()):
            s.append( str(child) )
        return "\n".join(s)

    def parse(self):
        '''
        examine all the vgroup nodes
        and reconstruct the root of the tree
        and the root (global) attributes
        and the children (all at once)

        note: if there any datasets in the root node, we miss them now
        '''
        vgroups = {}
        ref = -1
        while 1:

            # walk through all the vgroups by ref (a.k.a. id)
            try:
                ref = self.v.getid(ref)
            except pyhdf.HDF.HDF4Error,_:    # no more vgroup
                break

            # obtain vg and vgroup
            if ref in vgroups:      # already seen this one
                vgroup = vgroups[ref]
                vg = vgroup.obj
            else:                   # make a new vgroup at the root level
                vg = self.v.attach(ref)
                if vg._class in SPECIAL_CLASSES:    # ignore these
                    continue
                vgroups[ref] = vgroup = HDF4_Vgroup(self, vg)
                self.children[vgroup.name] = vgroup

            # examine all the child nodes
            for tag, tref in vg.tagrefs():
                if vg.inqtagref(tag, tref):
                    if tag == pyhdf.HDF.HC.DFTAG_NDG:
                        sds = self.sd.select( self.sd.reftoindex(tref) )
                        vgroup.children[tref] = HDF4_Dataset(vgroup, sds)
                    elif tref in vgroups:
                        vgroup.children[tref] = vgroups[tref]
                    elif tag == pyhdf.HDF.HC.DFTAG_VG:
                        vgroup.children[tref] = HDF4_Vgroup(vgroup, self.v.attach(tref))
                        vgroups[tref] = vgroup.children[tref]

        # gather any global (root) attributes
        for name, data in self.sd.attributes(full=True).items():
            if name not in SPECIAL_ATTRIBUTES:
                self.attr[name] = HDF4_Attribute(self, name=name,
                                                 value=data[0], hdftype=data[1])


class HDF4_Vgroup(HDF4_Master):
    '''
    vgroup object in HDF4 file
    '''
    children = {}   #: dictionary of children: key: object
    parent = None   #: object that contains this one

    def __init__(self, parent, vg):
        HDF4_Master.__init__(self)
        self.children = {}
        self.obj = vg
        self.name = vg._name
        self.parent = parent
        self.structure = STRUCTURE_TYPE["vgroup"]
        if parent is not None:
            self.root_obj = parent.root_obj
            delim = {True: "", False: "/"}[parent.path.endswith("/")]
            self.path = parent.path + delim + self.name

        for name, data in vg.attrinfo().items():
            if name not in SPECIAL_ATTRIBUTES:
                self.attr[name] = HDF4_Attribute(self, name=name,
                                             value=data[2], hdftype=data[0])

    def __str__(self):
        s = [self.path]
        for attr in sorted(self.attr.values()):
            s.append( str(attr) )
        for child in sorted(self.children.values()):
            s.append( str(child) )
        return "\n".join(s)


class HDF4_Dataset(HDF4_Master):
    '''
    SD object in HDF4 file
    '''
    parent = None   #: object that contains this one
    value = object

    def __init__(self, parent, sds):
        HDF4_Master.__init__(self)
        self.obj = sds
        data = sds.info()
        self.name = data[0]
        self.hdftype = data[3]      #: one of the SDC.xxx values
        self.parent = parent
        self.structure = STRUCTURE_TYPE["SDS"]
        if parent is not None:
            self.root_obj = parent.root_obj
            delim = {True: "", False: "/"}[parent.path.endswith("/")]
            self.path = parent.path + delim + self.name

        for name, data in sds.attributes(full=True).items():
            if name not in SPECIAL_ATTRIBUTES:
                self.attr[name] = HDF4_Attribute(self, name=name,
                                                 value=data[0], hdftype=data[1])

    def __str__(self):
        data = self.obj.info()
        if data[1] == 1:
            shape = ""
        else:
            shape = data[2]
        s = ["%s: %s %s" % (self.path, SDC_TYPE[self.hdftype], str(shape))]
        for attr in sorted(self.attr.values()):
            s.append( str(attr) )
        return "\n".join(s)


class HDF4_Attribute(HDF4_Master):
    '''
    attribute object in HDF4 file
    '''

    def __init__(self, parent, name=None, value=None, hdftype=None):
        HDF4_Master.__init__(self)
        if name is None or value is None or hdftype is None:
            raise SyntaxError, "most provide values for all three arguments: name, value, hdftype"
        self.obj = {'name':name, 'value':value, 'hdftype':hdftype}
        self.name = name
        self.value = value
        self.hdftype = hdftype
        self.parent = parent
        self.structure = STRUCTURE_TYPE["attribute"]
        if parent is not None:
            self.root_obj = parent.root_obj
            delim = {True: "@", False: "/@"}[parent.path.endswith("/")]
            self.path = parent.path + delim + self.name

    def __str__(self):
        return "%s = %s" % ( self.path, str(self.value) )


import os
import random
import javalang  # Importing the javalang library for parsing Java code.
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node  # Node class from javalang to represent elements of the AST.
import torch  # PyTorch library for tensor operations.
from anytree import AnyNode, RenderTree  # anytree library to create and visualize tree structures.
from anytree import find
from createclone_java import getedge_nextsib, getedge_flow, getedge_nextstmt, getedge_nexttoken, getedge_nextuse

def get_token(node):
    """
    Extracts the token from a node in the AST. A token could be a string, a modifier, or a node type.
    """
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token

def get_child(root):
    """
    Retrieves the children of a given AST node. Expands any nested lists within the children.
    """
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        """ A helper function to flatten nested lists within the AST nodes. """
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    return list(expand(children))

def get_sequence(node, sequence):
    """
    Traverses the AST from a given node, appending each token to the sequence.
    """
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node, nodelist):
    """
    Collects all nodes in the AST starting from a given node.
    """
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child, nodelist)

def createtree(root, node, nodelist, parent=None):
    """
    Creates a tree structure from AST nodes.
    """
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)



def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    """
    Extracts nodes and edges from the AST for a given node, focusing only on AST structure.
    """
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)

def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    """
    Extracts nodes, edges, and edge types from the AST for a given node.
    """
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

def countnodes(node,ifcount,whilecount,forcount,blockcount):
    """
    Counts the occurrences of specific node types in the AST.
    """
    token=node.token
    if token=='IfStatement':
        ifcount+=1
    if token=='WhileStatement':
        whilecount+=1
    if token=='ForStatement':
        forcount+=1
    if token=='BlockStatement':
        blockcount+=1
    print(ifcount,whilecount,forcount,blockcount)
    for child in node.children:
        countnodes(child,ifcount,whilecount,forcount,blockcount)
'''def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([1])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)'''

def createast():
    """
    Creates ASTs from Java source files located in a specified directory.
    """
    asts=[]
    paths=[]
    alltokens=[]
    dirname = 'BCB/bigclonebenchdata/'
    for rt, dirs, files in os.walk(dirname):
        for file in files:
            programfile=open(os.path.join(rt,file),encoding='utf-8')
            #print(os.path.join(rt,file))
            programtext=programfile.read()
            #programtext=programtext.replace('\r','')
            programtokens=javalang.tokenizer.tokenize(programtext)
            #print(list(programtokens))
            parser=javalang.parse.Parser(programtokens)
            programast=parser.parse_member_declaration()
            paths.append(os.path.join(rt,file))
            asts.append(programast)
            get_sequence(programast,alltokens)
            programfile.close()
            #print(programast)
            #print(alltokens)
    astdict=dict(zip(paths,asts))
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount = 0
    switchcount = 0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print(ifcount,whilecount,forcount,blockcount,docount,switchcount)
    print('allnodes ',len(alltokens))
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    return astdict,vocabsize,vocabdict

def createseparategraph(astdict,vocablen,vocabdict,device,mode='astonly',nextsib=False,ifedge=False,whileedge=False,foredge=False,blockedge=False,nexttoken=False,nextuse=False):
    """
    Creates separate graphs for each AST in the dictionary.
    """
    pathlist=[]
    treelist=[]
    # print('ifedge ',ifedge)
    # print('whileedge ',whileedge)
    # print('foredge ',foredge)
    # print('blockedge ',blockedge)
    # print('nexttoken', nexttoken)
    # print('nextuse ',nextuse)
    # print(len(astdict)
    for path,tree in astdict.items():
        #print(tree)
        #print(path)
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)
        #print(path)
        #print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        return
        #x = torch.tensor(x, dtype=torch.long, device=device)
        edge_index=[edgesrc, edgetgt]
        #edge_index = torch.tensor([edgesrc, edgetgt], dtype=torch.long, device=device)
        astlength=len(x)
        #print(x)
        #print(edge_index)
        #print(edge_attr)
        pathlist.append(path)
        treelist.append([[x,edge_index,edge_attr],astlength])
        astdict[path]=[[x,edge_index,edge_attr],astlength]
    #treedict=dict(zip(pathlist,treelist))
    #print(totalif,totalwhile,totalfor,totalblock)
    return astdict

def creategmndata(id,treedict,vocablen,vocabdict,device):
    """
    Creates data for Graph Matching Networks (GMN) from AST trees.
    """
    indexdir='BCB/'
    if id=='0':
        trainfile = open(indexdir+'traindata.txt')
        validfile = open(indexdir+'devdata.txt')
        testfile = open(indexdir+'testdata.txt')
    elif id=='11':
        trainfile = open(indexdir+'traindata11.txt')
        validfile = open(indexdir+'devdata.txt')
        testfile = open(indexdir+'testdata.txt')
    else:
        print('file not exist')
        quit()
    trainlist=trainfile.readlines()
    validlist=validfile.readlines()
    testlist=testfile.readlines()
    traindata=[]
    validdata=[]
    testdata=[]
    print('train data')
    traindata=createpairdata(treedict,trainlist,device=device)
    print('valid data')
    validdata=createpairdata(treedict,validlist,device=device)
    print('test data')
    testdata=createpairdata(treedict,testlist,device=device)
    return traindata, validdata, testdata

def createpairdata(treedict,pathlist,device):
    """
    Creates data for Graph Matching Networks (GMN) from AST trees.
    """
    datalist=[]
    countlines=1
    for line in pathlist:
        #print(countlines)
        countlines += 1
        pairinfo = line.split()
        code1path='BCB'+pairinfo[0].strip('.')
        code2path='BCB'+pairinfo[1].strip('.')
        label=int(pairinfo[2])
        data1 = treedict[code1path]
        data2 = treedict[code2path]
        x1,edge_index1,edge_attr1,ast1length=data1[0][0],data1[0][1],data1[0][2],data1[1]
        x2,edge_index2,edge_attr2,ast2length=data2[0][0],data2[0][1],data2[0][2],data2[1]
        '''matchsrc = []
        matchtgt = []
        for i in range(ast1length):
            for j in range(ast2length):
                matchsrc.append(i)
                matchtgt.append(j)
        match_index=[matchsrc, matchtgt]'''
        #match_index = torch.tensor([matchsrc, matchtgt], dtype=torch.long, device=device)
        if edge_attr1==[]:
            edge_attr1 = None
            edge_attr2 = None
        data = [[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2], label]
        datalist.append(data)
    return datalist
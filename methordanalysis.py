import ast
import copy

class CodeInputer:
    def __init__(self):
        self.code = ""
        self.ret = ""
    def Set(self, code):
        self.code = code
    def Get(self):
        return self.code
    def Ret(self, ret=None):
        if ret is None: return self.ret
        else: self.ret = ret     

class MAI:
    def __init__(self):
        self.ConstantTableEquals = [0,0.09,0.58,0.9,1.12,1.24,1.32,1.41,1.42,1.49, 1.51, 1.53, 1.56, 1.57, 1.59]
        self.NameOfAlternative = []
        self.NormalizedVectorsOfMainMatrix = []
        self.NormalaizedVectorsForParameters = {}
        self.InterfaceVar = {"_UI": None}

    def GetAnwser(self, is_alts):
        more_info = []
        response = []
        for x in range(is_alts):
            response.append(0)
            more_info.append([])
            #self.NormalaizedVectorsForParameters[vec][0][x]
            for ind, vec in enumerate(self.NormalaizedVectorsForParameters):
                response[x] += self.NormalaizedVectorsForParameters[vec][0][x] * self.NormalizedVectorsOfMainMatrix[self.NormalaizedVectorsForParameters[vec][1]]
                more_info[x].append([self.NormalaizedVectorsForParameters[vec][0][x] * self.NormalizedVectorsOfMainMatrix[self.NormalaizedVectorsForParameters[vec][1]], self.NormalaizedVectorsForParameters[vec][1]])
        return [response, more_info]

    def ConsistencyIndex(self, Lmax, argCounts):
        return (Lmax-argCounts)/(argCounts-1)

    def TakeConsistencyRelation(self, ConsistencyIndex, members):
        return (ConsistencyIndex/self.ConstantTableEquals[members-1])*100

    def AnalysisTable(self, table): #[[1.0, 0.5, 0.3333333], [2.0, 1.0, 3.0], [3.0, 0.3333333, 1.0]]
        VectorTo = []
        for row in table:
            Vector = 1
            for parametr in row:
                Vector*=parametr
            VectorTo.append(Vector**(1/len(row)))
        
        SumVectors = sum(VectorTo)
        NormalizedVectors = list([vector/SumVectors for vector in VectorTo])
        RotatedTable = []
        for y in range(len(table)):
            RotatedTable.append([])
            for x in range(len(table[y])):
                RotatedTable[y].append(table[x][y])

        SumColumns = []
        for column in RotatedTable:
            SumColumns.append(sum(column))

        MulSumAndVector = list([SumColumns[x]*NormalizedVectors[x] for x in range(len(SumColumns))])
        LMax = sum(MulSumAndVector)
        ConsistencyIndex = self.ConsistencyIndex(LMax, len(MulSumAndVector))
        ConsistencyRelation = self.TakeConsistencyRelation(ConsistencyIndex, len(MulSumAndVector))
        return [NormalizedVectors, ConsistencyRelation]
    
    def convertExpr2Expression(self, Expr):
        Expr.lineno = 0
        Expr.col_offset = 0
        result = ast.Expression(Expr.value, lineno=0, col_offset = 0)
        return result

    def exec_with_return(self, code):

        code_ast = ast.parse(code)

        init_ast = copy.deepcopy(code_ast)
        init_ast.body = code_ast.body[:-1]

        last_ast = copy.deepcopy(code_ast)
        last_ast.body = code_ast.body[-1:]
        variables = self.InterfaceVar|globals()
        exec(compile(init_ast, "<ast>", "exec"), variables)
        if type(last_ast.body[0]) == ast.Expr:
            return eval(compile(self.convertExpr2Expression(last_ast.body[0]), "<ast>", "eval"), variables)
        else:
            exec(compile(last_ast, "<ast>", "exec"), variables)
"""LLMCL Parser - Parses contract specifications into AST."""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from .ast_nodes import (
    ContractNode, RequireNode, EnsureNode, EnsureProbNode,
    TemporalNode, ExpressionNode, BinaryOpNode, UnaryOpNode,
    FunctionCallNode, IdentifierNode, LiteralNode, AttributeAccessNode,
    ImportNode, CompositionNode, ConditionalNode,
    ContractPriority, ConflictResolution, TemporalOperator
)


class LLMCLParser:
    """Parser for LLM Contract Language."""
    
    def __init__(self):
        self.tokens = []
        self.current = 0
        self.source = ""
        self.line = 1
        self.column = 1
    
    def parse(self, source: str) -> ContractNode:
        """Parse LLMCL source code into AST."""
        self.source = source
        self.tokens = self._tokenize(source)
        self.current = 0
        return self._parse_contract()
    
    def parse_expression(self, expr_str: str) -> ExpressionNode:
        """Parse a single expression."""
        self.source = expr_str
        self.tokens = self._tokenize(expr_str)
        self.current = 0
        return self._parse_expression()
    
    def _tokenize(self, source: str) -> List[Dict[str, Any]]:
        """Tokenize the source code."""
        # Token patterns
        patterns = [
            # Keywords
            (r'\bcontract\b', 'CONTRACT'),
            (r'\brequire\b', 'REQUIRE'),
            (r'\bensure\b', 'ENSURE'),
            (r'\bensure_prob\b', 'ENSURE_PROB'),
            (r'\btemporal\b', 'TEMPORAL'),
            (r'\bimport\b', 'IMPORT'),
            (r'\bfrom\b', 'FROM'),
            (r'\bas\b', 'AS'),
            (r'\bif\b', 'IF'),
            (r'\belse\b', 'ELSE'),
            (r'\band\b', 'AND'),
            (r'\bor\b', 'OR'),
            (r'\bnot\b', 'NOT'),
            (r'\bin\b', 'IN'),
            (r'\btrue\b', 'TRUE'),
            (r'\bfalse\b', 'FALSE'),
            (r'\bnull\b', 'NULL'),
            # Temporal operators
            (r'\balways\b', 'ALWAYS'),
            (r'\beventually\b', 'EVENTUALLY'),
            (r'\bnext\b', 'NEXT'),
            (r'\buntil\b', 'UNTIL'),
            (r'\bsince\b', 'SINCE'),
            (r'\bwithin\b', 'WITHIN'),
            # Priority levels
            (r'\bcritical\b', 'CRITICAL'),
            (r'\bhigh\b', 'HIGH'),
            (r'\bmedium\b', 'MEDIUM'),
            (r'\blow\b', 'LOW'),
            # Conflict resolution
            (r'\bfirst_wins\b', 'FIRST_WINS'),
            (r'\blast_wins\b', 'LAST_WINS'),
            (r'\bmost_restrictive\b', 'MOST_RESTRICTIVE'),
            (r'\bleast_restrictive\b', 'LEAST_RESTRICTIVE'),
            (r'\bmerge\b', 'MERGE'),
            (r'\bfail_on_conflict\b', 'FAIL_ON_CONFLICT'),
            # Literals
            (r'\d+\.\d+', 'FLOAT'),
            (r'\d+', 'INT'),
            (r'"([^"\\\\]|\\\\.)*"|\'([^\'\\\\]|\\\\.)*\'', 'STRING'),
            # Identifiers
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
            # Operators
            (r'==', 'EQ'),
            (r'!=', 'NEQ'),
            (r'<=', 'LEQ'),
            (r'>=', 'GEQ'),
            (r'<', 'LT'),
            (r'>', 'GT'),
            (r'\+', 'PLUS'),
            (r'-', 'MINUS'),
            (r'\*', 'STAR'),
            (r'/', 'SLASH'),
            (r'%', 'PERCENT'),
            (r'=', 'ASSIGN'),
            # Delimiters
            (r'\(', 'LPAREN'),
            (r'\)', 'RPAREN'),
            (r'\{', 'LBRACE'),
            (r'\}', 'RBRACE'),
            (r'\[', 'LBRACKET'),
            (r'\]', 'RBRACKET'),
            (r',', 'COMMA'),
            (r'\.', 'DOT'),
            (r':', 'COLON'),
            (r';', 'SEMICOLON'),
            (r'@', 'AT'),
            # Comments and whitespace
            (r'#[^\n]*', 'COMMENT'),
            (r'\s+', 'WHITESPACE'),
        ]
        
        # Compile patterns
        token_re = '|'.join(f'(?P<{name}>{pattern})' for pattern, name in patterns)
        token_pattern = re.compile(token_re)
        
        tokens = []
        line = 1
        column = 1
        
        for match in token_pattern.finditer(source):
            kind = match.lastgroup
            value = match.group()
            
            if kind == 'WHITESPACE':
                # Update position but don't create token
                if '\n' in value:
                    line += value.count('\n')
                    column = len(value.split('\n')[-1]) + 1
                else:
                    column += len(value)
            elif kind != 'COMMENT':
                tokens.append({
                    'type': kind,
                    'value': value,
                    'line': line,
                    'column': column
                })
                column += len(value)
        
        tokens.append({'type': 'EOF', 'value': '', 'line': line, 'column': column})
        return tokens
    
    def _current_token(self) -> Dict[str, Any]:
        """Get current token."""
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return self.tokens[-1]  # EOF
    
    def _advance(self) -> Dict[str, Any]:
        """Advance to next token."""
        token = self._current_token()
        if token['type'] != 'EOF':
            self.current += 1
        return token
    
    def _match(self, *types: str) -> bool:
        """Check if current token matches any of the given types."""
        return self._current_token()['type'] in types
    
    def _consume(self, token_type: str, message: str) -> Dict[str, Any]:
        """Consume a token of the given type or raise error."""
        token = self._current_token()
        if token['type'] == token_type:
            return self._advance()
        raise SyntaxError(f"{message} at line {token['line']}, column {token['column']}")
    
    def _parse_contract(self) -> ContractNode:
        """Parse a complete contract."""
        # Parse imports
        imports = []
        while self._match('IMPORT', 'FROM'):
            imports.append(self._parse_import())
        
        # Parse contract declaration
        self._consume('CONTRACT', "Expected 'contract' keyword")
        name_token = self._consume('IDENTIFIER', "Expected contract name")
        
        # Parse optional metadata
        priority = ContractPriority.DEFAULT
        conflict_resolution = ConflictResolution.FAIL_ON_CONFLICT
        description = None
        
        if self._match('LPAREN'):
            self._advance()
            # Parse contract options
            while not self._match('RPAREN'):
                if self._match('IDENTIFIER'):
                    key = self._advance()['value']
                    self._consume('ASSIGN', "Expected '=' after option name")
                    
                    if key == 'priority':
                        priority = self._parse_priority()
                    elif key == 'conflict_resolution':
                        conflict_resolution = self._parse_conflict_resolution()
                    elif key == 'description':
                        desc_token = self._consume('STRING', "Expected string for description")
                        description = desc_token['value'][1:-1]  # Remove quotes
                    
                    if self._match('COMMA'):
                        self._advance()
                
            self._consume('RPAREN', "Expected ')' after contract options")
        
        self._consume('LBRACE', "Expected '{' after contract declaration")
        
        # Parse contract body
        requires = []
        ensures = []
        temporal = []
        
        while not self._match('RBRACE'):
            if self._match('REQUIRE'):
                requires.append(self._parse_require())
            elif self._match('ENSURE_PROB'):
                ensures.append(self._parse_ensure_prob())
            elif self._match('ENSURE'):
                ensures.append(self._parse_ensure())
            elif self._match('TEMPORAL'):
                temporal.append(self._parse_temporal())
            else:
                raise SyntaxError(f"Unexpected token: {self._current_token()['value']}")
        
        self._consume('RBRACE', "Expected '}' after contract body")
        
        return ContractNode(
            name=name_token['value'],
            description=description,
            priority=priority,
            conflict_resolution=conflict_resolution,
            requires=requires,
            ensures=ensures,
            temporal=temporal,
            imports=imports,
            line=name_token['line'],
            column=name_token['column']
        )
    
    def _parse_import(self) -> ImportNode:
        """Parse import statement."""
        if self._match('FROM'):
            self._advance()
            module = self._consume('IDENTIFIER', "Expected module name")['value']
            self._consume('IMPORT', "Expected 'import' after module name")
            
            contracts = []
            if self._match('STAR'):
                self._advance()
            else:
                contracts.append(self._consume('IDENTIFIER', "Expected contract name")['value'])
                while self._match('COMMA'):
                    self._advance()
                    contracts.append(self._consume('IDENTIFIER', "Expected contract name")['value'])
            
            alias = None
            if self._match('AS'):
                self._advance()
                alias = self._consume('IDENTIFIER', "Expected alias name")['value']
            
            return ImportNode(module=module, contracts=contracts, alias=alias)
        else:
            # import module
            self._advance()  # consume 'import'
            module = self._consume('IDENTIFIER', "Expected module name")['value']
            return ImportNode(module=module)
    
    def _parse_priority(self) -> ContractPriority:
        """Parse priority level."""
        if self._match('CRITICAL'):
            self._advance()
            return ContractPriority.CRITICAL
        elif self._match('HIGH'):
            self._advance()
            return ContractPriority.HIGH
        elif self._match('MEDIUM'):
            self._advance()
            return ContractPriority.MEDIUM
        elif self._match('LOW'):
            self._advance()
            return ContractPriority.LOW
        else:
            raise SyntaxError("Expected priority level (critical, high, medium, low)")
    
    def _parse_conflict_resolution(self) -> ConflictResolution:
        """Parse conflict resolution strategy."""
        if self._match('FIRST_WINS'):
            self._advance()
            return ConflictResolution.FIRST_WINS
        elif self._match('LAST_WINS'):
            self._advance()
            return ConflictResolution.LAST_WINS
        elif self._match('MOST_RESTRICTIVE'):
            self._advance()
            return ConflictResolution.MOST_RESTRICTIVE
        elif self._match('LEAST_RESTRICTIVE'):
            self._advance()
            return ConflictResolution.LEAST_RESTRICTIVE
        elif self._match('MERGE'):
            self._advance()
            return ConflictResolution.MERGE
        elif self._match('FAIL_ON_CONFLICT'):
            self._advance()
            return ConflictResolution.FAIL_ON_CONFLICT
        else:
            raise SyntaxError("Expected conflict resolution strategy")
    
    def _parse_require(self) -> RequireNode:
        """Parse require statement."""
        token = self._advance()  # consume 'require'
        condition = self._parse_expression()
        
        message = None
        severity = "error"
        auto_fix = None
        tags = []
        
        # Parse optional clauses
        while self._match('IDENTIFIER'):
            clause = self._current_token()['value']
            if clause == 'message':
                self._advance()
                self._consume('COLON', "Expected ':' after 'message'")
                message_token = self._consume('STRING', "Expected string message")
                message = message_token['value'][1:-1]  # Remove quotes
            elif clause == 'severity':
                self._advance()
                self._consume('COLON', "Expected ':' after 'severity'")
                severity = self._consume('IDENTIFIER', "Expected severity level")['value']
            elif clause == 'auto_fix':
                self._advance()
                self._consume('COLON', "Expected ':' after 'auto_fix'")
                auto_fix = self._parse_expression()
            elif clause == 'tags':
                self._advance()
                self._consume('COLON', "Expected ':' after 'tags'")
                tags = self._parse_string_list()
            else:
                break
        
        return RequireNode(
            condition=condition,
            message=message,
            severity=severity,
            auto_fix=auto_fix,
            tags=tags,
            line=token['line'],
            column=token['column']
        )
    
    def _parse_ensure(self) -> EnsureNode:
        """Parse ensure statement."""
        token = self._advance()  # consume 'ensure'
        condition = self._parse_expression()
        
        message = None
        severity = "error"
        auto_fix = None
        tags = []
        
        # Parse optional clauses
        while self._match('IDENTIFIER'):
            clause = self._current_token()['value']
            if clause == 'message':
                self._advance()
                self._consume('COLON', "Expected ':' after 'message'")
                message_token = self._consume('STRING', "Expected string message")
                message = message_token['value'][1:-1]  # Remove quotes
            elif clause == 'severity':
                self._advance()
                self._consume('COLON', "Expected ':' after 'severity'")
                severity = self._consume('IDENTIFIER', "Expected severity level")['value']
            elif clause == 'auto_fix':
                self._advance()
                self._consume('COLON', "Expected ':' after 'auto_fix'")
                auto_fix = self._parse_expression()
            elif clause == 'tags':
                self._advance()
                self._consume('COLON', "Expected ':' after 'tags'")
                tags = self._parse_string_list()
            else:
                break
        
        return EnsureNode(
            condition=condition,
            message=message,
            severity=severity,
            auto_fix=auto_fix,
            tags=tags,
            line=token['line'],
            column=token['column']
        )
    
    def _parse_ensure_prob(self) -> EnsureProbNode:
        """Parse ensure_prob statement."""
        token = self._advance()  # consume 'ensure_prob'
        condition = self._parse_expression()
        self._consume('COMMA', "Expected ',' after condition")
        
        prob_token = self._current_token()
        if prob_token['type'] == 'FLOAT':
            probability = float(self._advance()['value'])
        elif prob_token['type'] == 'INT':
            probability = float(self._advance()['value'])
        else:
            raise SyntaxError("Expected probability value")
        
        message = None
        window_size = 100
        tags = []
        
        # Parse optional clauses
        while self._match('IDENTIFIER'):
            clause = self._current_token()['value']
            if clause == 'message':
                self._advance()
                self._consume('COLON', "Expected ':' after 'message'")
                message_token = self._consume('STRING', "Expected string message")
                message = message_token['value'][1:-1]
            elif clause == 'window_size':
                self._advance()
                self._consume('COLON', "Expected ':' after 'window_size'")
                window_size = int(self._consume('INT', "Expected integer window size")['value'])
            elif clause == 'tags':
                self._advance()
                self._consume('COLON', "Expected ':' after 'tags'")
                tags = self._parse_string_list()
            else:
                break
        
        return EnsureProbNode(
            condition=condition,
            probability=probability,
            message=message,
            window_size=window_size,
            tags=tags,
            line=token['line'],
            column=token['column']
        )
    
    def _parse_temporal(self) -> TemporalNode:
        """Parse temporal constraint."""
        token = self._advance()  # consume 'temporal'
        
        # Parse temporal operator
        if self._match('ALWAYS'):
            operator = TemporalOperator.ALWAYS
            self._advance()
        elif self._match('EVENTUALLY'):
            operator = TemporalOperator.EVENTUALLY
            self._advance()
        elif self._match('NEXT'):
            operator = TemporalOperator.NEXT
            self._advance()
        elif self._match('UNTIL'):
            operator = TemporalOperator.UNTIL
            self._advance()
        elif self._match('SINCE'):
            operator = TemporalOperator.SINCE
            self._advance()
        elif self._match('WITHIN'):
            operator = TemporalOperator.WITHIN
            self._advance()
            scope = int(self._consume('INT', "Expected integer scope for 'within'")['value'])
        else:
            raise SyntaxError("Expected temporal operator")
        
        condition = self._parse_expression()
        
        message = None
        tags = []
        
        # Parse optional clauses
        while self._match('IDENTIFIER'):
            clause = self._current_token()['value']
            if clause == 'message':
                self._advance()
                self._consume('COLON', "Expected ':' after 'message'")
                message_token = self._consume('STRING', "Expected string message")
                message = message_token['value'][1:-1]
            elif clause == 'tags':
                self._advance()
                self._consume('COLON', "Expected ':' after 'tags'")
                tags = self._parse_string_list()
            else:
                break
        
        return TemporalNode(
            operator=operator,
            condition=condition,
            message=message,
            scope=scope if operator == TemporalOperator.WITHIN else None,
            tags=tags,
            line=token['line'],
            column=token['column']
        )
    
    def _parse_expression(self) -> ExpressionNode:
        """Parse expression with precedence climbing."""
        return self._parse_or()
    
    def _parse_or(self) -> ExpressionNode:
        """Parse OR expression."""
        left = self._parse_and()
        
        while self._match('OR'):
            op = self._advance()
            right = self._parse_and()
            left = BinaryOpNode('or', left, right, line=op['line'], column=op['column'])
        
        return left
    
    def _parse_and(self) -> ExpressionNode:
        """Parse AND expression."""
        left = self._parse_not()
        
        while self._match('AND'):
            op = self._advance()
            right = self._parse_not()
            left = BinaryOpNode('and', left, right, line=op['line'], column=op['column'])
        
        return left
    
    def _parse_not(self) -> ExpressionNode:
        """Parse NOT expression."""
        if self._match('NOT'):
            op = self._advance()
            operand = self._parse_not()
            return UnaryOpNode('not', operand, line=op['line'], column=op['column'])
        
        return self._parse_comparison()
    
    def _parse_comparison(self) -> ExpressionNode:
        """Parse comparison expression."""
        left = self._parse_additive()
        
        while self._match('EQ', 'NEQ', 'LT', 'GT', 'LEQ', 'GEQ', 'IN'):
            op = self._advance()
            right = self._parse_additive()
            left = BinaryOpNode(op['value'], left, right, line=op['line'], column=op['column'])
        
        return left
    
    def _parse_additive(self) -> ExpressionNode:
        """Parse addition/subtraction."""
        left = self._parse_multiplicative()
        
        while self._match('PLUS', 'MINUS'):
            op = self._advance()
            right = self._parse_multiplicative()
            left = BinaryOpNode(op['value'], left, right, line=op['line'], column=op['column'])
        
        return left
    
    def _parse_multiplicative(self) -> ExpressionNode:
        """Parse multiplication/division."""
        left = self._parse_postfix()
        
        while self._match('STAR', 'SLASH', 'PERCENT'):
            op = self._advance()
            right = self._parse_postfix()
            left = BinaryOpNode(op['value'], left, right, line=op['line'], column=op['column'])
        
        return left
    
    def _parse_postfix(self) -> ExpressionNode:
        """Parse postfix expressions (attribute access, function calls)."""
        expr = self._parse_primary()
        
        while True:
            if self._match('DOT'):
                self._advance()
                attr = self._consume('IDENTIFIER', "Expected attribute name")
                expr = AttributeAccessNode(expr, attr['value'], line=attr['line'], column=attr['column'])
            elif self._match('LPAREN'):
                self._advance()
                args, kwargs = self._parse_arguments()
                self._consume('RPAREN', "Expected ')' after arguments")
                if isinstance(expr, IdentifierNode):
                    expr = FunctionCallNode(expr.name, args, kwargs, line=expr.line, column=expr.column)
                else:
                    raise SyntaxError("Invalid function call")
            else:
                break
        
        return expr
    
    def _parse_primary(self) -> ExpressionNode:
        """Parse primary expressions."""
        token = self._current_token()
        
        if self._match('TRUE'):
            self._advance()
            return LiteralNode(True, line=token['line'], column=token['column'])
        
        if self._match('FALSE'):
            self._advance()
            return LiteralNode(False, line=token['line'], column=token['column'])
        
        if self._match('NULL'):
            self._advance()
            return LiteralNode(None, line=token['line'], column=token['column'])
        
        if self._match('INT'):
            self._advance()
            return LiteralNode(int(token['value']), line=token['line'], column=token['column'])
        
        if self._match('FLOAT'):
            self._advance()
            return LiteralNode(float(token['value']), line=token['line'], column=token['column'])
        
        if self._match('STRING'):
            self._advance()
            # Remove quotes
            value = token['value'][1:-1]
            # Unescape string
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
            return LiteralNode(value, line=token['line'], column=token['column'])
        
        if self._match('IDENTIFIER'):
            self._advance()
            return IdentifierNode(token['value'], line=token['line'], column=token['column'])
        
        if self._match('LPAREN'):
            self._advance()
            expr = self._parse_expression()
            self._consume('RPAREN', "Expected ')' after expression")
            return expr
        
        raise SyntaxError(f"Unexpected token: {token['value']}")
    
    def _parse_arguments(self) -> Tuple[List[ExpressionNode], Dict[str, ExpressionNode]]:
        """Parse function arguments."""
        args = []
        kwargs = {}
        
        if not self._match('RPAREN'):
            # Parse first argument
            if self._match('IDENTIFIER') and self.tokens[self.current + 1]['type'] == 'ASSIGN':
                # Keyword argument
                name = self._advance()['value']
                self._consume('ASSIGN', "Expected '='")
                kwargs[name] = self._parse_expression()
            else:
                # Positional argument
                args.append(self._parse_expression())
            
            # Parse remaining arguments
            while self._match('COMMA'):
                self._advance()
                if self._match('IDENTIFIER') and self.tokens[self.current + 1]['type'] == 'ASSIGN':
                    # Keyword argument
                    name = self._advance()['value']
                    self._consume('ASSIGN', "Expected '='")
                    kwargs[name] = self._parse_expression()
                else:
                    # Positional argument
                    if kwargs:
                        raise SyntaxError("Positional arguments cannot follow keyword arguments")
                    args.append(self._parse_expression())
        
        return args, kwargs
    
    def _parse_string_list(self) -> List[str]:
        """Parse a list of strings."""
        strings = []
        self._consume('LBRACKET', "Expected '[' to start list")
        
        if not self._match('RBRACKET'):
            token = self._consume('STRING', "Expected string in list")
            strings.append(token['value'][1:-1])
            
            while self._match('COMMA'):
                self._advance()
                token = self._consume('STRING', "Expected string in list")
                strings.append(token['value'][1:-1])
        
        self._consume('RBRACKET', "Expected ']' to end list")
        return strings
'''
sql_parser.py

由 bigquery_view_parser.py 改写而来, 定义了需要用到的 sql 语法

'''
# bigquery_view_parser.py
#
# A parser to extract table names from BigQuery view definitions.
# This is based on the `select_parser.py` sample in pyparsing:
# https://github.com/pyparsing/pyparsing/blob/master/examples/select_parser.py
#
# Michael Smedberg
#
import sys
from pyparsing import *

sys.setrecursionlimit(3000)

ParserElement.enablePackrat()

class SQLParser:
    """Parser to extract table info from BigQuery view definitions

    Based on the BNF and examples posted at
    https://cloud.google.com/bigquery/docs/reference/legacy-sql
    """

    _parser = None
    _table_identifiers = set()
    _with_aliases = set()

    def get_table_names(self, sql_stmt):
        table_identifiers, with_aliases = self._parse(sql_stmt)

        # Table names and alias names might differ by case, but that's not
        # relevant- aliases are not case sensitive
        lower_aliases = SQLParser.lowercase_set_of_tuples(with_aliases)
        tables = {
            x
            for x in table_identifiers
            if not SQLParser.lowercase_of_tuple(x) in lower_aliases
        }

        # Table names ARE case sensitive as described at
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#case_sensitivity
        return tables

    def _parse(self, sql_stmt):
        SQLParser._table_identifiers.clear()
        SQLParser._with_aliases.clear()
        SQLParser._get_parser().parseString(sql_stmt, parseAll=True)

        return SQLParser._table_identifiers, SQLParser._with_aliases

    @classmethod
    def lowercase_of_tuple(cls, tuple_to_lowercase):
        return tuple(x.lower() if x else None for x in tuple_to_lowercase)

    @classmethod
    def lowercase_set_of_tuples(cls, set_of_tuples):
        return {SQLParser.lowercase_of_tuple(x) for x in set_of_tuples}

    @classmethod
    def _get_parser(cls):
        if cls._parser is not None:
            return cls._parser

        ParserElement.enablePackrat()

        LPAR, RPAR, COMMA = map(Suppress, "(),")
        APOS, DOT, SEMI, COLON = map(Suppress, "'.;:")
        ungrouped_select_stmt = Forward().setName("select statement")

        QUOTED_APOS = QuotedString("'")

        # fmt: off
        # keywords
        (
            SELECT, FROM, WHERE, DISTINCT,
            AND, OR, NOT, IN, BETWEEN, IS, NULL,
            INNER, LEFT, RIGHT, JOIN, ON, AS, GROUP, ORDER, BY, LIMIT, HAVING, ASC, DESC,
            NOW, CURDATE, ADDDATE, ADDTIME, DATE, TIMEDIFF,
            AVG, MAX, MIN, SUM, COUNT,
            LENGTH, ISNULL,
        ) = map(
            CaselessKeyword,
            """
            SELECT, FROM, WHERE, DISTINCT,
            AND, OR, NOT, IN, BETWEEN, IS, NULL,
            INNER, LEFT, RIGHT, JOIN, ON, AS, GROUP, ORDER, BY, LIMIT, HAVING, ASC, DESC,
            NOW, CURDATE, ADDDATE, ADDTIME, DATE, TIMEDIFF,
            AVG, MAX, MIN, SUM, COUNT,
            LENGTH, ISNULL,
            """.replace(",", "").split(),
        )

        keyword_nonfunctions = MatchFirst(
            (SELECT, FROM, WHERE, DISTINCT,
             AND, OR, NOT, IN, BETWEEN, IS, NULL,
             INNER, LEFT, RIGHT, JOIN, ON, AS, GROUP, ORDER, BY, LIMIT, HAVING, ASC, DESC,
            )
        )

        keyword = keyword_nonfunctions | MatchFirst(
            (NOW, CURDATE, ADDDATE, ADDTIME, DATE, TIMEDIFF,
             AVG, MAX, MIN, SUM, COUNT,
             LENGTH, ISNULL,
            )
        )

        # fmt: on

        identifier_word = Word(alphas + "_@#", alphanums + "@$#_")
        identifier = ~keyword + identifier_word.copy()
        # NOTE: Column names can be keywords.  Doc says they cannot, but in practice it seems to work.
        column_name = identifier.copy()
        qualified_column_name = Combine(
            column_name + ("." + column_name)[..., 6], adjacent=False
        )
        # NOTE: As with column names, column aliases can be keywords, e.g. functions like `current_time`.  Other
        # keywords, e.g. `from` make parsing pretty difficult (e.g. "SELECT a from from b" is confusing.)
        table_name = identifier.copy()
        table_alias = identifier.copy()

        # expression
        expr = Forward().setName("expression")

        integer = Regex(r"[+-]?\s?\d+")
        numeric_literal = Regex(r"[+-]?\d*\.?\d+([eE][+-]?\d+)?")
        string_literal = QUOTED_APOS
        literal_value = (
            numeric_literal
            | string_literal
            | NULL
            | NOW + LPAR + RPAR
            | CURDATE + LPAR + RPAR
        )

        grouping_term = expr.copy()
        ordering_term = Group(
            expr("order_key")
            + Optional(ASC | DESC)("direction")
        )("ordering_term")

        function_arg = expr.copy()("function_arg")
        function_args = Optional(
            "*"
            | Optional(DISTINCT)
            + delimitedList(function_arg)
        )("function_args")

        aggregate_function_name = (
            AVG
            | COUNT
            | MAX
            | MIN
            | SUM
        )
        analytic_function_name = (
            aggregate_function_name
        )("analytic_function_name")
        analytic_function = (
            analytic_function_name
            + LPAR
            + function_args
            + RPAR
        )("analytic_function")
        datetime_extractors = (
            DATE
        )

        expr_term = (
            (analytic_function)("analytic_function")
            | (LPAR + ungrouped_select_stmt + RPAR)("subselect")
            | (literal_value)("literal")
            | (datetime_extractors + LPAR + expr + RPAR)("datetime_extration")
            | (ADDTIME + LPAR + expr + COMMA + APOS + integer + COLON + integer + APOS + RPAR)("addtime")
            | (ADDDATE + LPAR + expr + COMMA + integer + RPAR)("adddate")
            | (TIMEDIFF + LPAR + expr + COMMA + expr + RPAR)("timediff")
            | (LENGTH + LPAR + expr + RPAR)("length")
            | (ISNULL + LPAR + expr + RPAR)("isnull")
            | qualified_column_name("column")
        )

        struct_term = LPAR + delimitedList(expr_term) + RPAR

        UNARY, BINARY, TERNARY = 1, 2, 3
        expr <<= infixNotation(
            (expr_term | struct_term),
            [
                (oneOf("- +") | NOT, UNARY, opAssoc.RIGHT),
                (oneOf("+ -"), BINARY, opAssoc.LEFT),
                (oneOf("= > < >= <= !="), BINARY, opAssoc.LEFT),
                (
                    IS + Optional(NOT)
                    | Optional(NOT) + IN,
                    BINARY,
                    opAssoc.LEFT,
                ),
                ((BETWEEN, AND), TERNARY, opAssoc.LEFT),
                (
                    Optional(NOT)
                    + IN
                    + LPAR
                    + Group(ungrouped_select_stmt | delimitedList(expr))
                    + RPAR,
                    UNARY,
                    opAssoc.LEFT,
                ),
                (AND, BINARY, opAssoc.LEFT),
                (OR, BINARY, opAssoc.LEFT),
            ],
        )

        join_constraint = Group(
            Optional(
                ON + expr
            )
        )("join_constraint")

        join_op = (
            COMMA
            | Group(
                Optional(
                    INNER
                    | LEFT
                    | RIGHT
                )
                + JOIN
            )
        )("join_op")

        join_source = Forward()

        def record_table_identifier(t):
            identifier_list = t.asList()
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._table_identifiers.add(tuple(padded_list))

        standard_table_part = ~keyword + Word(alphanums + "_")
        table_parts_identifier = (
            Optional(
                standard_table_part("project") + DOT
            )
            + Optional(
                standard_table_part("dataset") + DOT
            )
            + standard_table_part("table")
        ).setParseAction(record_table_identifier)

        table_identifier = (table_parts_identifier).setName("table_identifier")
        single_source = (
            (
                table_identifier
            )("index")
            | (LPAR + ungrouped_select_stmt + RPAR)
            | (LPAR + join_source + RPAR)
        ) + Optional(Optional(AS) + table_alias("table_alias"))

        join_source <<= single_source + (join_op + single_source + join_constraint)[...]

        result_column = Optional(table_name + ".") + "*" | expr

        ungrouped_select_no_with = (
            SELECT
            + Optional(DISTINCT)
            + Group(
                delimitedList(
                    (~FROM  + result_column),
                    allow_trailing_delim=True,
                )
            )("columns")
            + Optional(FROM + join_source("from*"))
            + Optional(WHERE + expr)
            + Optional(
                GROUP + BY + Group(delimitedList(grouping_term))("group_by_terms")
            )
            + Optional(HAVING + expr("having_expr"))
            + Optional(
                ORDER + BY + Group(delimitedList(ordering_term))("order_by_terms")
            )
        )
        select_no_with = ungrouped_select_no_with | (
            LPAR + ungrouped_select_no_with + RPAR
        )
        grouped_select_core = select_no_with | (LPAR + select_no_with + RPAR)

        ungrouped_select_stmt <<= (
            grouped_select_core
            + Optional(
                LIMIT
                + (Group(expr + COMMA + expr) | expr)(
                    "limit"
                )
            )
        )("select")
        select_stmt = (
            ungrouped_select_stmt | (LPAR + ungrouped_select_stmt + RPAR)
        ) + Optional(SEMI)

        # define comment format, and ignore them
        sql_comment = oneOf("-- #") + restOfLine | cStyleComment
        select_stmt.ignore(sql_comment)

        cls._parser = select_stmt
        return cls._parser


def predict_next_vocab(parser, vocab, sentence, eos_token):
    next_vocab = []
    # 词汇间加入空格, 除非是初始词汇
    if len(sentence) != 0:
        sentence = sentence + ' '
    curr_len = len(sentence)
    for word in vocab:
        # 在 type(parser) 是 pyparsing.core.And 的情况下, 
        # And 需要匹配多个字段,
        # 因此可能 success 为 False, 但 allResults 中的 loc 进度被推进了,
        success, allResults = parser.runTests(sentence + word, parseAll=False)
        if success or allResults[0][1].loc > curr_len:
            next_vocab.append(word)
    # 如果 parseAll=True 时能 parse 成功, 那 <eos> 就可以是潜在词汇
    success, _ = parser.runTests(sentence)
    if success and sentence:
        next_vocab.append(eos_token)
    return next_vocab


if __name__ == "__main__":
    parser = SQLParser._get_parser()

    ## parse file
    # test_file_path = 'a_dataset/dataset/data_train.sql'
    # with open(test_file_path, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         success, _ = parser.runTests(line)
    #         if not success:
    #             print("\n{}".format("OK" if success else "FAIL"))

    ## parse sentence
    test_sql = """
        select t_rt_info.Tag_code from t_rt_info left join ( select * from t_rt_data where date( t_rt_data.Result_time ) = '2023-08-31'  ) as t1 on t_rt_info.Tag_code = t1.Tag_code group by t_rt_info.Tag_code having count( * ) != 0 ;
    """

    success, a = parser.runTests(test_sql, parseAll=False)
    print("\n{}".format("OK" if success else "FAIL"))
from abc import abstractmethod, ABC
from json import load
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union, Any, List, Callable
from enum import Enum
from collections.abc import MutableSequence


class Type(Enum):
    Float = 0
    String = 1


def to_float(obj) -> float:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return float(obj) if obj is not None else None


def to_str(obj) -> str:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return str(obj) if obj is not None else None


def common(iterable): # from ChatGPT
    """
    returns True if all items of iterable are the same.
    :param iterable:
    :return:
    """
    try:
        # Nejprve zkusíme získat první prvek iterátoru
        iterator = iter(iterable)
        first_value = next(iterator)
    except StopIteration:
        # Vyvolá výjimku, pokud je iterátor prázdný
        raise ValueError("Iterable is empty")

    # Kontrola, zda jsou všechny další prvky stejné jako první prvek
    for value in iterator:
        if value != first_value:
            raise ValueError("Not all values are the same")

    # Vrací hodnotu, pokud všechny prvky jsou stejné
    return first_value


class Column(MutableSequence):# implement MutableSequence (some method are mixed from abc)
    """
    Representation of column of dataframe. Column has datatype: float columns contains
    only floats and None values, string columns contains strings and None values.
    """
    def __init__(self, data: Iterable, dtype: Type):
        self.dtype = dtype
        self._cast = to_float if self.dtype == Type.Float else to_str
        # cast function (it casts to floats for Float datatype or
        # to strings for String datattype)
        self._data = [self._cast(value) for value in data]

    def __len__(self) -> int:
        """
        Implementation of abstract base class `MutableSequence`.
        :return: number of rows
        """
        return len(self._data)

    def __getitem__(self, item: Union[int, slice]) -> Union[float,
                                    str, list[str], list[float]]:
        """
        Indexed getter (get value from index or sliced sublist for slice).
        Implementation of abstract base class `MutableSequence`.
        :param item: index or slice
        :return: item or list of items
        """
        return self._data[item]

    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        """
        Indexed setter (set value to index, or list to sliced column)
        Implementation of abstract base class `MutableSequence`.
        :param key: index or slice
        :param value: simple value or list of values

        """
        self._data[key] = self._cast(value)

    def append(self, item: Any) -> None:
        """
        Item is appended to column (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param item: appended value
        """
        self._data.append(self._cast(item))

    def insert(self, index: int, value: Any) -> None:
        """
        Item is inserted to colum at index `index` (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param index:  index of new item
        :param value:  inserted value
        :return:
        """
        self._data.insert(index, self._cast(value))

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Remove item from index `index` or sublist defined by `slice`.
        :param index: index or slice
        """
        del self._data[index]

    def permute(self, indices: List[int]) -> 'Column':
        """
        Return new column which items are defined by list of indices (to original column).
        (eg. `Column(["a", "b", "c"]).permute([0,0,2])`
        returns  `Column(["a", "a", "c"])
        :param indices: list of indexes (ints between 0 and len(self) - 1)
        :return: new column
        """
        #assert len(indices) == len(self)
        new_data = [self._data[i] for i in indices] # using comprehension for reindexing column by pater (indice)

        return Column(new_data, self.dtype)

    def copy(self) -> 'Column':
        """
        Return shallow copy of column.
        :return: new column with the same items
        """
        # FIXME: value is cast to the same type (minor optimisation problem)
        return Column(self._data, self.dtype)
    
    def is_float(self) -> bool:
        """
        returns True if its float
        """
        if self.dtype == Type.Float:
            return True
        return False
    
    def get_data(self):
        return self.copy()

    def get_formatted_item(self, index: int, *, width: int):
        """
        Auxiliary method for formating column items to string with `width`
        characters. Numbers (floats) are right aligned and strings left aligned.
        Nones are formatted as aligned "n/a".
        :param index: index of item
        :param width:  width
        :return:
        """
        assert width > 0
        if self._data[index] is None:
            if self.dtype == Type.Float:
                return "n/a".rjust(width)
            else:
                return "n/a".ljust(width)
        return format(self._data[index],
                      f"{width}s" if self.dtype == Type.String else f"-{width}.2g")

class DataFrame:
    """
    Dataframe with typed and named columns
    """
    def __init__(self, columns: Dict[str, Column]):
        """
        :param columns: columns of dataframe (key: name of dataframe),
                        lengths of all columns has to be the same
        """
        assert len(columns) > 0, "Dataframe without columns is not supported"
        self._size = common(len(column) for column in columns.values())
        # deep copy od dict `columns`
        self._columns = {name: column.copy() for name, column in columns.items()}

    def __getitem__(self, index: int) -> Tuple[Union[str,float]]:
        """
        Indexed getter returns row of dataframe as tuple
        :param index: index of row
        :return: tuple of items in row
        """
        assert index >= 0 and index <= self._size
        rlist = []
        for col_name in self._columns:
            rlist.append(self._columns[col_name][index])
        return tuple(rlist)

    def __iter__(self) -> Iterator[Tuple[Union[str, float]]]:
        """
        :return: iterator over rows of dataframe
        """
        for i in range(len(self)):
            yield tuple(c[i] for c in self._columns.values())

    def __len__(self) -> int:
        """
        :return: count of rows
        """
        return self._size

    @property
    def columns(self) -> Iterable[str]:
        """
        :return: names of columns (as iterable object)
        """
        return self._columns.keys()

    def __repr__(self) -> str:
        """
        :return: string representation of dataframe (table with aligned columns)
        """
        lines = []
        lines.append(" ".join(f"{name:12s}" for name in self.columns))
        for i in range(len(self)):
            lines.append(" ".join(self._columns[cname].get_formatted_item(i, width=12)
                                     for cname in self.columns))
        return "\n".join(lines)

    def append_column(self, col_name:str, column: Column) -> None:
        """
        Appends new column to dataframe (its name has to be unique).
        :param col_name:  name of new column
        :param column: data of new column
        """
        if col_name in self._columns:
            raise ValueError("Duplicate column name")
        self._columns[col_name] = column.copy()

    def append_row(self, row: Iterable) -> None:
        """
        Appends new row to dataframe.
        :param row: tuple of values for all columns
        """
        if len(row) != len(self._columns): # Checking for same lenght of row and number of columns
            # raising exception if lenght of row is not same as number of columns
            raise ValueError(f"Lenght of append data isn`t matching number of collumns -> {len(self._columns)}")
        i:int = 0 # indexer for row !possible tuning!
        for col_name in self.columns:
            self._columns[col_name].append(row[i]) # appending new values
            i += 1
        self._size += 1 # after finaliying of append we have increase size

    def filter(self, col_name:str,
               predicate: Callable[[Union[float, str]], bool]) -> 'DataFrame':
        """
        Returns new dataframe with rows which values in column `col_name` returns
        True in function `predicate`.

        :param col_name: name of tested column
        :param predicate: testing function
        :return: new dataframe
        """
        ...

    def unique(self, col_name) -> 'DataFrame':
        if col_name not in self._columns:
            raise NameError(f"column {col_name} is not in Dataframe")
        unique_val = set()
        indices = []

        for i, val in enumerate(self._columns[col_name]):
            if val not in unique_val:
                unique_val.add(val)
                indices.append(i)
        new_cols = {column: self._columns[column].permute(indices) for column in self.columns}
        return DataFrame(new_cols)

    def sample():
        ...

    def sort(self, col_name:str, ascending=True) -> 'DataFrame':
        """
        Sort dataframe by column with `col_name` ascending or descending.
        :param col_name: name of key column
        :param ascending: direction of sorting
        :return: new dataframe
        """
        if col_name not in self._columns:
            raise NameError(f"column {col_name} is not in Dataframe")
        
        sorted_indices = sorted(
        range(len(self)),
        key=lambda i: (self._columns[col_name][i] is None, self._columns[col_name][i]),
        reverse=not ascending
        )
        sorted_columns = {name: self._columns[name].permute(sorted_indices) for name in self.columns}
        return DataFrame(sorted_columns)

    def avg(self, col_name):
        """
        returns average value in column if Type is string
        """
        # checking for existence of column
        if col_name not in self._columns:
            raise NameError(f"column {col_name} is not in Dataframe")
        # if column is type float calculate average value
        if self._columns[col_name].is_float():
            data_pointer = self._columns[col_name].get_data()
            for i in range(self._size):
                if data_pointer[i] == None:
                    data_pointer[i] = 0.0
            ravg = sum(data_pointer) / len(self._columns[col_name])
            return ravg
        else:
            raise TypeError(f"Collumn {col_name} is not float type") # exception in case column is nto float
    
    def max(self, col_name):
        """
        returns maximal value in column
        """
        if col_name not in self._columns:
            raise NameError(f"column {col_name} is not in Dataframe")   
        
        data_pointer = [x for x in self._columns[col_name].get_data() if x is not None] # deleting None values
        rmax = max(data_pointer)
        return rmax
    
    def min(self, col_name):
        """
        returns maximal value in column
        """
        if col_name not in self._columns:
            raise NameError(f"column {col_name} is not in Dataframe")   
        
        data_pointer = [x for x in self._columns[col_name].get_data() if x is not None] # deleting None values
        rmin = min(data_pointer)
        return rmin
    
    def describe(self) -> str:
        """
        similar to pandas but only with min, max and avg statistics for floats and count"
        :return: string with formatted decription
        """
        lines = [] # list for lines
        for col_name in self.columns:
            col = self._columns[col_name]
            if self._columns[col_name].is_float(): # statistics for float column
                col_min = self.min(col_name)
                col_max = self.max(col_name)
                col_avg = self.avg(col_name)
                ncount = len([v for v in col if v is not None]) # gets count of not None values
                lines.append(f"{col_name:12s} type: float min: {col_min:.2f} max: {col_max:.2f} avg: {col_avg:.2f} count: {ncount}")
            else: # statistics for string column
                ncount = len([v for v in col if v is not None]) # gets count of not None values
                lines.append(f"{col_name:12s} type: str count: {ncount}")
        return "\n".join(lines)

    def inner_join(self, other: 'DataFrame', self_key_column: str,
                   other_key_column: str) -> 'DataFrame':
        """
            Inner join between self and other dataframe with join predicate
            `self.key_column == other.key_column`.

            Possible collision of column identifiers is resolved by prefixing `_other` to
            columns from `other` data table.
        """
        ...

    def setvalue(self, col_name: str, row_index: int, value: Any) -> None:
        """
        Set new value in dataframe.
        :param col_name:  name of culumns
        :param row_index: index of row
        :param value:  new value (value is cast to type of column)
        :return:
        """
        col = self._columns[col_name]
        col[row_index] = col._cast(value)

    @staticmethod
    def read_csv(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by CSV reader
        """
        return CSVReader(path).read()

    @staticmethod
    def read_json(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by JSON reader
        """
        return JSONReader(path).read()


class Reader(ABC):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

    @abstractmethod
    def read(self) -> DataFrame:
        raise NotImplemented("Abstract method")


class JSONReader(Reader):
    """
    Factory class for creation of dataframe by CSV file. CSV file must contain
    header line with names of columns.
    The type of columns should be inferred from types of their values (columns which
    contains only value has to be floats columns otherwise string columns),
    """
    def read(self) -> DataFrame:
        with open(self.path, "rt") as f:
            json = load(f)
        columns = {}
        for cname in json.keys(): # cyklus přes sloupce (= atributy JSON objektu)
            dtype = Type.Float if all(value is None or isinstance(value, Real)
                                      for value in json[cname]) else Type.String
            columns[cname] = Column(json[cname], dtype)
        return DataFrame(columns)


class CSVReader(Reader):
    """
    Factory class for creation of dataframe by JSON file. JSON file must contain
    one object with attributes which array values represents columns.
    The type of columns are inferred from types of their values (columns which
    contains only value is floats columns otherwise string columns),
    """
    def read(self) -> 'DataFrame':
        ...


if __name__ == "__main__":
    df = DataFrame(
        dict(
        a=Column([None, 3.1415], Type.Float),
        b=Column(["a", 2], Type.String),
        c=Column(range(2), Type.Float)
        )
        )
    df.setvalue("a", 1, 42)
    #print(df)

    #df = DataFrame.read_json("data.json")
    #print(df)
    alfa = (10,20)
    #df.append_row(alfa)
    for i in range(1, 11):
        df.append_row((i,i+i,i*i))
        df.append_row((1,i,i))
    #print(df)
    print(df)
    print()
    #print(df.describe())
    nd = df.unique("a")
    print(nd.sort("a"))
###
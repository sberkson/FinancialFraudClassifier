##############################################
# Programmer: Ben Puryear
# Class: CptS 322-02, Spring 2022
# Programming Assignment #3
# 2/22/2022
# I did not attempt the bonus...
#
# Description: This file contains the class MyPyTable, which is an alternative way to store data created by me for this PA.
##############################################
import copy
import csv
from tabulate import tabulate


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        cols = len(self.column_names)
        return rows, cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if (
            col_identifier.isnumeric()
        ):  # check to see if the col_identifier is an int (the index) or the string (the label)
            column_index = int(col_identifier)
        else:
            try:
                column_index = self.column_names.index(col_identifier)
            except ValueError:  # throw the ValueError if the col_identifier is not a valid column name
                raise ValueError(
                    "Invalid column identifier: " + col_identifier)

        col_values = []  # this will be the returned list of values

        for row in self.data:
            if (
                include_missing_values
            ):  # check to see if the include_missing_values flag is active
                col_values.append(row[column_index])
            else:  # if not ...
                if row[column_index] != "NA":  # ignore the NA
                    col_values.append(row[column_index])

        return col_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for col in range(len(row)):
                try:  # uses a try/except block to see if the value can be converted to a float
                    row[col] = float(row[col])
                except ValueError:  # valueerror is raised if the value cannot be converted to a float
                    # print("Could not convert " + row[col] + " to a float")
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_data = []  # creates a new list that will become the new data
        for i in range(len(self.data)):  # iterates through the data
            # checks to see if the index is in the list of indexes to drop
            if i not in row_indexes_to_drop:
                # if not, add the row to the new data
                new_data.append(self.data[i])
        self.data = new_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")
        # creates the reader object from the csv module
        reader = csv.reader(infile)
        self.column_names = next(reader)
        self.data = []  # initialize the new data list
        for row in reader:
            self.data.append(row)  # adds the row from the reader to the data

        # calls the convert_to_numeric method on the newley created data table
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")

        # creates the writer module from the csv module
        writer = csv.writer(
            outfile, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL
        )  # found this on stackoverflow
        # writes the column names (header) to the file
        writer.writerow(self.column_names)
        for row in self.data:
            writer.writerow(row)  # writes the row to the file

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dupe_list = []  # creates a list to store the indexes of the duplicates

        # this will be a temporary list that will include all of the itterated through rows
        # (will be used as the check against) (will not have the dupes)
        occured_already = []

        for i in range(len(self.data)):
            # make the new temporary 1D list that is just the key_column_names
            list_temp = []
            for col_name in key_column_names:
                index = self.column_names.index(col_name)
                list_temp.append(self.data[i][index])

            # if the temporary list that is only the key_column_names is in the list, it is a dupe
            if list_temp in occured_already:
                dupe_list.append(i)
            else:
                occured_already.append(list_temp)

        return dupe_list

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        new_data = (
            []
        )  # a new list that self.data will be set to after the rest of the function

        for row in self.data:
            row_clean = True  # this will be a flag that is originally true, but will be set to false if there is ever an "NA" val in the row
            for val in row:
                if val == "NA":  # this is the check
                    row_clean = False  # and this sets the flag to false
            # if it checked every value in the row and the flag is still true...
            if row_clean:
                new_data.append(row)  # add that row to the new data

        self.data = new_data  # replace the self.data with the new data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # first lets get the index for the col_name
        col_name_index = self.column_names.index(col_name)

        # now we are going to calculate the average of col_name
        col_vals = []
        for i in range(len(self.data)):
            if self.data[i][col_name_index] != "NA":
                col_vals.append(self.data[i][col_name_index])
        col_vals_average = sum(col_vals) / len(col_vals)

        # now that we have the col_name_average we can go through the data list again and replace all NA with it
        for i in range(len(self.data)):
            if self.data[i][col_name_index] == "NA":
                self.data[i][col_name_index] = col_vals_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_headers = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []
        for col in col_names:  # this is going to go through each of the passed in names
            # initialize a new list that has one value (the name)
            new_list = [col]
            # now we can use get_col()
            new_col = self.get_column(col)
            # check to see if it is empty
            if len(new_col) == 0:
                return MyPyTable()

            # compute the stats
            new_min = min(new_col)
            new_max = max(new_col)
            new_mid = (new_max + new_min) / 2
            new_avg = sum(new_col) / len(new_col)
            new_col.sort()  # needed for calculating the median
            if len(new_col) % 2 == 0:  # it is even
                new_median = (
                    new_col[(len(new_col) - 1) // 2] +
                    new_col[(len(new_col) // 2)]
                ) / 2
            else:
                new_median = new_col[(len(new_col) // 2)]

            new_list.append(new_min)
            new_list.append(new_max)
            new_list.append(new_mid)
            new_list.append(new_avg)
            new_list.append(new_median)

            # now append new_list to stats data
            stats_data.append(new_list)

        new_table = MyPyTable(stats_headers, stats_data)
        return new_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        new_col_names = []
        for col_name in self.column_names:
            new_col_names.append(col_name)
        for col_name in other_table.column_names:
            if col_name not in new_col_names:
                new_col_names.append(col_name)

        new_data = []

        for row in range(len(self.data)):  # for every row in self.data
            # we are going to compare it to every row in other_table (NOTE: not efficient)
            for other_row in range(len(other_table.data)):
                valid_row = True  # this a bool that will be used later
                # we are going through all of the key_column_names to check the values of the row against all other_table rows
                for i in range(len(key_column_names)):
                    current_self_key_col_value = self.data[row][
                        self.column_names.index(key_column_names[i])
                    ]
                    current_other_key_col_value = other_table.data[other_row][
                        other_table.column_names.index(key_column_names[i])
                    ]

                    # if there is ever a point where the key_col values do not match, set valid_row to valse
                    if current_self_key_col_value != current_other_key_col_value:
                        valid_row = False
                        break

                # NOTE: IMPORTANT, there can also be multiple valid rows.
                if valid_row:
                    new_row = []
                    # we are going through each of the columns that will be in the resulting table
                    for col_name in new_col_names:
                        # if the col_name is in self.column_names
                        if col_name in self.column_names:
                            # we add the row to the new_row
                            new_row.append(
                                self.data[row][self.column_names.index(
                                    col_name)]
                            )
                        else:  # otherwise it must be a other_table column exlusive
                            new_row.append(
                                other_table.data[other_row][
                                    other_table.column_names.index(col_name)
                                ]
                            )  # so we add it
                    # now that the new_row is complete, we add it to the new_data
                    new_data.append(new_row)

        new_table = MyPyTable(new_col_names, new_data)

        return new_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with

            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        new_col_names = []
        for col_name in self.column_names:
            new_col_names.append(col_name)
        for col_name in other_table.column_names:
            if col_name not in new_col_names:
                new_col_names.append(col_name)

        new_data = []

        for row in range(len(self.data)):  # for every row in self.data
            # we are first going to add everything from self.data to new_data
            new_row = []
            already_added_a_row = False
            # we are going through each of the columns that will be in the resulting table
            for col_name in new_col_names:
                if col_name in self.column_names:  # if the col_name is in self.column_names
                    # we add the row to the new_row
                    new_row.append(
                        self.data[row][self.column_names.index(col_name)])
                else:  # otherwise it must be a other_table column exlusive
                    # so we add it (temporary, will be changed if there is valud data)
                    new_row.append("NA")
                    for row2 in other_table.data:
                        valid_row = True  # this a bool that will be used later
                        for new_col_name in new_col_names:
                            if new_col_name in other_table.column_names:
                                if new_row[new_col_names.index(new_col_name)] != "NA":
                                    if new_row[new_col_names.index(new_col_name)] != row2[other_table.column_names.index(new_col_name)]:
                                        valid_row = False
                                        break
                        if valid_row:
                            newest_row = new_row.copy()
                            newest_row[new_col_names.index(
                                col_name)] = row2[other_table.column_names.index(col_name)]
                            new_data.append(newest_row)
                            already_added_a_row = True

                    if new_row not in new_data and already_added_a_row == False:
                        new_data.append(new_row)

        for i in range(len(other_table.data)):
            seen_before = False
            for j in range(len(new_data)):
                valid_row = True
                for k in range(len(key_column_names)):
                    current_self_key_col_value = other_table.data[i][other_table.column_names.index(
                        key_column_names[k])]
                    current_other_key_col_value = new_data[j][new_col_names.index(
                        key_column_names[k])]
                    if current_self_key_col_value != current_other_key_col_value:
                        valid_row = False
                        break
                if valid_row:
                    seen_before = True
                    break
            if not seen_before:
                new_row = []
                for col_name in new_col_names:
                    if col_name in other_table.column_names:
                        new_row.append(
                            other_table.data[i][other_table.column_names.index(col_name)])
                    else:
                        new_row.append("NA")
                new_data.append(new_row)

        new_table = MyPyTable(new_col_names, new_data)

        return new_table

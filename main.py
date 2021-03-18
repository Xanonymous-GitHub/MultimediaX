import sys

PACKAGE_ENTRY_POINT_NAME = 'run'
PACKAGE_NAME_PREFIX = 'd'


class Importer:

    def __init__(self, name: str, subpackage_serial: str):
        # name is the package name but we don't sure that it exists.
        self.name = name

        # because we don't sure that the package is actually exists, so we set the target package to None.
        self.package = None

        # user may provide a subpackage_serial.
        self.subpackage_serial = subpackage_serial

    def __enter__(self):
        try:
            # try to load the package using the name provided when constructing.
            self.package = __import__(PACKAGE_NAME_PREFIX + str(self.name))
        except ModuleNotFoundError:
            pass  # we ignore this exception because it will be handled when we call the 'run' method.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # module resources will be automatically released after used.

    def __str__(self):
        return str(self.name)

    def run(self):
        # if package is truly exists,
        if self.package:
            # if user provide the sub package entry function suffix serial number,
            if self.subpackage_serial:
                try:
                    # try to call the sub package entry function.
                    getattr(self.package, PACKAGE_ENTRY_POINT_NAME + '_' + self.subpackage_serial)()
                except AttributeError:
                    # the sub package entry function not found.
                    print('package {0} entry function with serial number {1} can not be found.'
                          .format(self.package.__name__, self.subpackage_serial))

            # if user NOT provide the sub package entry function suffix serial number,
            else:
                try:
                    # try to directly call the entry function.
                    getattr(self.package, PACKAGE_ENTRY_POINT_NAME)()
                except AttributeError:
                    # user may forget to rename the entry function when remove the other sub packages.
                    try:
                        # if the situation happened, default to call the entry function that has serial number 1.
                        getattr(self.package, PACKAGE_ENTRY_POINT_NAME + '_1')()
                    except AttributeError:
                        # that means user not implemented the entry function.
                        print('package {0} not have any recognizable entry functions.')

        else:
            print(self.name, 'not found.')


def main():
    # store the sub package entry function suffix serial number.
    subpackage_serial: str = str()

    # if using cli arguments to specify target function,
    if len(sys.argv) > 1:
        # get package name.
        package_name: str = sys.argv[1]

        # if this package have a sub package that needs to be run,
        if len(sys.argv) > 2:
            # get sub package entry function suffix serial number.
            subpackage_serial: str = sys.argv[2]

    # if cli arguments not specified, try to ask user after running.
    else:
        # get everything from user input and split into a list.
        package_request = input(
            'Please tell me which day you want to execute the project in class '
            '[MMDD [, SUB_PACKAGE_SERIAL]]?: '
        ).split()

        # if user provide the sub package entry function suffix serial number,
        if len(package_request) > 1:
            # get the package name and sub package entry function suffix serial number.
            package_name, subpackage_serial = package_request
        else:
            # get the package name only.
            package_name = package_request[0]

        # if user not provide package name, show error message.
        if not package_name:
            print('project not provided, aborted.')
            return

    # use initialized self-defined contextmanager 'Importer' to execute the target function.
    with Importer(package_name, subpackage_serial) as pkg:
        # call the pre-defined entry interface method of the contextmanager.
        pkg.run()

        # show the ended message.
        print(pkg, 'finished.')


if __name__ == '__main__':
    main()

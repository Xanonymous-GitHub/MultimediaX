import sys


class Import:
    def __init__(self, name):
        self.name = name
        self.package = None

    def __enter__(self):
        try:
            self.package = __import__('d' + str(self.name))
        except ModuleNotFoundError:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # module resources will be automatically released after used.

    def __str__(self):
        return str(self.name)

    def run(self):
        if self.package:
            self.package.run()
        else:
            print(self.name, 'not found.')


def main():
    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = input('Please tell me which day you want to execute the project in class [MMDD]?: ')
        if not package_name:
            print('project not provided, aborted.')
            return

    with Import(package_name) as pkg:
        pkg.run()
        print(pkg, 'finished.')


if __name__ == '__main__':
    main()

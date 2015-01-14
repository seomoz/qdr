clean:
	# Remove the build
	rm -rf build dist
	# And all of our pyc files
	rm -f qdr/*.pyc test/*.pyc
	# All compiled files
	rm -f qdr/*.so qdr/ranker.cpp
	# And lastly, .coverage files
	rm -f .coverage

test: nose

nose:
	rm -rf .coverage
	nosetests --exe --cover-package=qdr --with-coverage --cover-branches -v --cover-erase 

unittest:
	python -m unittest discover -s test

build: clean
	python setup.py build_ext --inplace

install: build
	python setup.py install

run:
	python3 main.py

test:
	python3 -m unittest discover . "*_test.py" -v

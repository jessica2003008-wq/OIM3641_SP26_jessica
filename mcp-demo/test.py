import httpx

test_cases = [
    {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
    {'sepal_length': 6.0, 'sepal_width': 2.7, 'petal_length': 5.1, 'petal_width': 1.6},
    {'sepal_length': 6.3, 'sepal_width': 3.3, 'petal_length': 6.0, 'petal_width': 2.5},
]

for test_case in test_cases:
    response = httpx.post('http://localhost:8000/predict', json=test_case)
    print(response.json())


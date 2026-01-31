python
import math

def calculate_loan_payment(interest, term, present_value):
    """
    Calculates the monthly payment for a loan.

    Args:
        interest (float): The annual interest rate (e.g., 0.05 for 5%).
        term (int): The loan term in years.
        present_value (float): The principal loan amount (present value).

    Returns:
        float: The calculated monthly loan payment.
    """
    # Convert annual interest rate to monthly
    monthly_interest_rate = interest / 12

    # Convert term in years to total number of payments
    num_payments = term * 12

    # Handle the case of zero interest rate
    if monthly_interest_rate == 0:
        if num_payments == 0:
            return 0.0 # Or raise an error, depending on desired behavior for a 0-term loan
        return present_value / num_payments

    # Calculate monthly payment using the standard formula:
    # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
    # where P = present_value, i = monthly_interest_rate, n = num_payments
    
    # Calculate (1 + i)^n
    pow_factor = math.pow(1 + monthly_interest_rate, num_payments)
    
    # Calculate the numerator: i * (1 + i)^n
    numerator = monthly_interest_rate * pow_factor
    
    # Calculate the denominator: (1 + i)^n - 1
    denominator = pow_factor - 1

    # Calculate the monthly payment
    monthly_payment = present_value * (numerator / denominator)

    return monthly_payment
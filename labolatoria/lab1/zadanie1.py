def prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def select_primes(x):
    return list(filter(prime, x))


print(select_primes([3, 6, 11, 25, 19]))

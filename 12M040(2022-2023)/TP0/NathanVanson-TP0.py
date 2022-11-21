### Nathan Vanson UNIGE ###

### License : GNU Public License (GPL)
### Course : Numerical Analysis (TP0)

# ========== LIBRAIRES ========== #

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========== WORK ========== #

# Exercice 1 :

# A 

def gauss(x):
    return np.exp(-x**2) / np.sqrt(np.pi)

def test_gauss():
    assert (
        np.abs(gauss(0.123) - 0.5557182026803765) < 1e-10
    ), "Erreur dans la gaussienne"
    assert np.abs(gauss(-10000)) < 1e-16, "Erreur dans la gaussienne"
    x = np.linspace(-10, 10, 23)
    y = gauss(x)
    assert y.shape == x.shape, "La fonction ne fonctionne pas pour les arrays"
    assert np.all(y >= 0), "La gaussienne doit être positive"
    
test_gauss()

# B

x = np.linspace(-3, 3, 100)
y = gauss(x)
#print(y)
plt.figure(figsize=(5, 3))
plt.plot(x, y, color="r", linestyle="-", label="Gaussian")
plt.title("Gaussian Function")
plt.legend()
plt.show()

# C

def gauss_integrale(a, b):
    resultat, erreur = quad(gauss, a, b)
    return resultat

def test_gauss_integrale():
    assert np.abs(gauss_integrale(0, 1) - 0.4213503964748575) < 1e-10
    assert np.abs(gauss_integrale(0, 100) - 0.5) < 1e-10


test_gauss_integrale()

# D 

print(gauss_integrale(-100, 100))
print(gauss_integrale(-np.inf, np.inf))

# The 2 results are identical but the cause for one is due to numerical errors

# Exercice 2 :

# A

def poly(coeff, x):
    resultat = 0
    n = len(coeff)
    for i in range(n):
        resultat += coeff[i] * x**(n-i-1)
    return resultat

def test_poly():
    assert (
        poly([0], 10) == 0
    ), "La fonction ne fonctionne pas pour le polynôme nul"
    assert (
        poly([], 10) == 0
    ), "La fonction ne fonctionne pas pour le polynôme nul"
    assert (
        poly([1.23], 10) == 1.23
    ), "La fonction ne fonctionne pas pour un polynôme constant"
    assert poly([1, 1, 0], 10) == 110, "Erreur dans la fonction poly"
    x = np.linspace(0, 1, 21)
    y = poly([1, 1, 0], x)
    assert y.shape == x.shape, "La fonction ne fonctionne pas pour les arrays"
    assert abs(np.sum(y) - 17.675) < 1e-10, "Erreur dans la fonction poly"


test_poly()

# B

def derivee(coeff):
    n = len(coeff)
    resultat = []
    for i in range(n-1):
        resultat.append(coeff[i] * (n - i - 1)) 
    return resultat

def test_derivee():
    assert (
        derivee([0]) == []
    ), "Derivee d'un polynome constant est nulle (coeff = [])"
    assert (
        derivee([10]) == []
    ), "Derivee d'un polynome constant est nulle (coeff = [])"
    assert derivee([1, 2, 3, 4]) == [3, 4, 3], "Erreur dans la fonction derivee"
    coeff_correct = [500, 0, 0, 0, 0]
    assert (
        derivee([100, 0, 0, 0, 0, 0]) == coeff_correct
    ), "Erreur dans la fonction derivee"

test_derivee()

# C

def methode_newton(coeff, x0, n_pas):
    resultat = [x0]
    for i in range(n_pas):
        x = resultat[-1]
        resultat.append(x - poly(coeff, x) / poly(derivee(coeff), x))
    return np.array(resultat)

def test_methode_newton():
    # poly(x) = x**2 - 2
    coeff = [1, 0, -2]
    x_newton = methode_newton(coeff, 1, 10)
    assert x_newton.shape == (
        11,
    ), "La méthode de Newton ne retourne pas un array de la bonne taille"
    assert (
        x_newton[0] == 1
    ), "La méthode de Newton ne retourne pas le bon point de départ"
    assert (
        np.abs(x_newton[-1] ** 2 - 2) < 1e-8
    ), "La méthode ne converge pas vers le point correct"

    x_newton = methode_newton(coeff, -1, 10)
    assert x_newton[-1] < 0, "La méthode semble ignorer le point de départ"
    assert (
        np.abs(x_newton[-1] ** 2 - 2) < 1e-8
    ), "La méthode ne converge pas vers le bon point"

    coeff = [12.3, 4.20, 1.79, 0.42, 0.01]
    x_newton = methode_newton(coeff, 1, 10)
    x_final = x_newton[-1]
    assert (
        np.abs(poly(coeff, x_final)) < 1e-8
    ), "La méthode ne fonctionne pas pour tous les polynômes"


test_methode_newton()

x = np.linspace(0, 2, 30)
coeff = [1, 0, -2]  # x**2 - 2
plt.axhline(0, c="k", ls="--", label="$y = 0$")
plt.plot(x, poly(coeff, x), c="b", label="$x^2 - 2$")

x_newton = methode_newton(coeff, 1, 10)
plt.plot(x_newton, poly(coeff, x_newton), "o", c="r", label="Méthode de Newton")
plt.legend()
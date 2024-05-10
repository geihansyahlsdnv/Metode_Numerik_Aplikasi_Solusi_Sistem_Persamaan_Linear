import numpy as np
import scipy.linalg as linalg

def solusi_persamaan_linear_crout(matriks_A, vektor_b):

  try:
    P, L, U = linalg.lu(matriks_A, permute_l=False)
    y = linalg.solve_triangular(L, vektor_b, lower=True)
    x = linalg.solve_triangular(U, y)
    return x
  except ValueError as e:
     print(f"Matriks A singular. Tidak bisa melakukan dekomposisi LU: {e}")
     return None

# Contoh penggunaan
matriks_A = np.array([[4, 2], [3, -3]])
vektor_b = np.array([8, 2])

solusi = solusi_persamaan_linear_crout(matriks_A, vektor_b)

if solusi is not None:
  print(f"Solusi Persamaan Linier dengan Metode Crout:\n{solusi}")
else:
  print("Gagal menyelesaikan persamaan linier. Matriks A singular.")

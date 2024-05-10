import numpy as np

def solve_persamaan_linear_invers(matriks_A, vektor_b):

  try:
    invers_matriks_A = np.linalg.inv(matriks_A)
    vektor_solusi = np.dot(invers_matriks_A, vektor_b)
    return vektor_solusi
  except np.linalg.LinAlgError as e:
      print(f"Matriks A singular. Invers tidak dapat dihitung: {e}")
      return None

#Contoh Penggunaan
matriks_A = np.array([[9, 4, 2], [2, -4, 1], [2, 6, 4]])
vektor_b = np.array([6, 4, 9])

solusi = solve_persamaan_linear_invers(matriks_A, vektor_b)

if solusi is not None:
  print("Solusi menggunakan metode invers matriks:", solusi)
else:
  print("Gagal menyelesaikan persamaan linier. Matriks A singular.")

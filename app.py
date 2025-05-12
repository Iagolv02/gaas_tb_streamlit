import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# Defino una función para asignar los hoppings de forma automática
def add_hoppings_slater_koster_gaas(model, orb_names, orb_coords, lat, cutoff, sk):
    '''
    model -- modelo de TB definido anteriormente para el material considerado.
    orb_names -- nombres de los orbitales que se involucrén en cada hopping.
    orb_coords -- posiciones de los orbitales involucrados.
    lat -- vectores de red asociados a la celda primitiva del material.
    cutoff -- distancia máxima hasta la cual se consideran interacciones mediante hopping.
    sk -- diccionario con los parámetros de slater-koster.
    '''
    
    lat = np.array(lat)
    natom = len(orb_coords)
    
    # Recorremos todos los pares de orbitales (i, j)
    for i in range(natom):
        r_i = np.array(orb_coords[i])
        orb_i = orb_names[i]
        
        for j in range(natom):
            if j <= i:
                continue  # Evita duplicar la interacción
            r_j = np.array(orb_coords[j])
            orb_j = orb_names[j]

            # Recorremos los diferentes vecinos de la celda primitiva desplazandonos con los vectores de red
            for ix in [-1,0,1]:
                for iy in [-1,0,1]:
                    for iz in [-1,0,1]:
                        shift_cart = ix*lat[0] + iy*lat[1] + iz*lat[2]  # Defino el desplazamiento en la red
                        dvec = shift_cart + (r_j - r_i) # Defino el vector que une los orbitales, teniendo en cuenta a que celda pertenecen
                        
                        # No tengo en cuenta la interacción si es la misma posición o si estoy más lejos de una distancia definida con cuttof
                        dist = np.linalg.norm(dvec)
                        if dist < 1e-6:
                            continue
                        if dist > cutoff:
                            continue
                        
                        # Calculo los cosenos directores
                        l = dvec[0]/dist
                        m = dvec[1]/dist
                        n = dvec[2]/dist
                        
                        # Defino estas funciones para identificar entre que orbitales son los hoppings
                        def is_s(name):
                            return name.startswith("s_")
                        def is_p(name):
                            return (name.startswith("p_"))
                        
                        tval = 0.0
                        
                        if is_s(orb_i) and is_s(orb_j): # Hoppings s-s
                            tval = sk["Vss_sigma"]
                            
                        elif is_s(orb_i) and is_p(orb_j): # Hoppings s-p
                            if orb_j.startswith("p_x"):
                                tval = l * sk["Vsp_sigma"]
                            elif orb_j.startswith("p_y"):
                                tval = m * sk["Vsp_sigma"]
                            elif orb_j.startswith("p_z"):
                                tval = n * sk["Vsp_sigma"]
                                
                        elif is_p(orb_i) and is_s(orb_j): # Hoppings p-s
                            if orb_i.startswith("p_x"):
                                tval = -l * sk["Vsp_sigma"]
                            elif orb_i.startswith("p_y"):
                                tval = -m * sk["Vsp_sigma"]
                            elif orb_i.startswith("p_z"):
                                tval = -n * sk["Vsp_sigma"]
                                
                        elif is_p(orb_i) and is_p(orb_j): # Hoppings p-p
    
                        # Identificamos que p_i son cada uno de los orbitales considerados
                            def p_idx(oname):
                                if   oname.startswith("p_x"): return 0
                                elif oname.startswith("p_y"): return 1
                                else:                         return 2
                            i1 = p_idx(orb_i)
                            i2 = p_idx(orb_j)
                            
                            # Defino los valores de los cosenos directores para realizar el cálculo
                            dir_cos = [l, m, n]
                            li = dir_cos[i1]
                            lj = dir_cos[i2]
                            
                            if i1 == i2:  # px px, py py, pz pz
                                tval = (li**2)*sk["Vpp_sigma"] + (1 - li**2)*sk["Vpp_pi"]
                            else:
                                # px py, px pz, py pz
                                tval = li*lj*(sk["Vpp_sigma"] - sk["Vpp_pi"])
                        else:
                            # Esto es por si no es s ni p
                            continue
                        
                        model.set_hop(tval, i, j, [ix,iy,iz])
                        
# Defino una función para construir el modelo TB dado los parámetros de hopping y las energías on-site
def build_gaas_model(sk_params, onsite_params):
    '''
    sk_params -- diccionario con los parámetros de slater-koster.
    onsite_params -- diccionario con las energías on-site.
    '''
    
    #Defino la celda del GaAs (Estructura del diamante)
    a = 5.653
    lat = [
        [0.5*a, 0.5*a, 0.0],
        [0.0,   0.5*a, 0.5*a],
        [0.5*a, 0.0,   0.5*a]
    ]
    
    # Defino los orbitales para las dos subredes (A en (0,0,0) y B en (0.25a,0.25a,0.25a))
    orb_coords = []
    orb_names  = []
    
    # Ga (Red A)
    A_pos = [0.0, 0.0, 0.0]
    A_orbs = ["s_Ga", "p_x_Ga", "p_y_Ga", "p_z_Ga"]
    for orb in A_orbs:
        orb_coords.append(A_pos)
        orb_names.append(orb)
        
    # As (Red B)
    B_pos = [0.25*a, 0.25*a, 0.25*a]
    B_orbs = ["s_As", "p_x_As", "p_y_As", "p_z_As"]
    for orb in B_orbs:
        orb_coords.append(B_pos)
        orb_names.append(orb)
    
    # Defino mi modelo de TB
    model = tb_model(3, 3, lat, orb_coords)
    
    # Asigno las energías on-site a cada orbital
    #   Para Ga: E_s_Ga y E_p_Ga
    #   Para As: E_s_As y E_p_As
    on_site = []
    for name in orb_names:
        if "_Ga" in name:
            if name.startswith("s_"):
                on_site.append(onsite_params["E_s_Ga"])
            else:
                on_site.append(onsite_params["E_p_Ga"])
        elif "_As" in name:
            if name.startswith("s_"):
                on_site.append(onsite_params["E_s_As"])
            else:
                on_site.append(onsite_params["E_p_As"])
    model.set_onsite(on_site)
    
    # Añado los hopping empleando la función anterior según los parámetros de Slater-Koster
    cutoff_dist = 3.8
    add_hoppings_slater_koster_gaas(model, orb_names, orb_coords, lat, cutoff_dist, sk_params)
    
    return model, orb_names, orb_coords, lat

# Parámetros iniciales
initial_Es_Ga = -4.94
initial_Ep_Ga = 3.11
initial_Es_As = -9.07
initial_Ep_As = 0.48
initial_Vss_sigma = -1.8
initial_Vsp_sigma = 2.0
initial_Vpp_sigma = 2.95
initial_Vpp_pi = -1.0

# Sliders de Streamlit
Es_Ga = st.slider('E_s_Ga', -10.0, 0.0, initial_Es_Ga, 0.1)
Ep_Ga = st.slider('E_p_Ga', 0.0, 10.0, initial_Ep_Ga, 0.1)
Es_As = st.slider('E_s_As', -15.0, -5.0, initial_Es_As, 0.1)
Ep_As = st.slider('E_p_As', -1.0, 5.0, initial_Ep_As, 0.1)
Vss_sigma = st.slider('Vss_sigma', -5.0, 0.0, initial_Vss_sigma, 0.1)
Vsp_sigma = st.slider('Vsp_sigma', 0.0, 5.0, initial_Vsp_sigma, 0.1)
Vpp_sigma = st.slider('Vpp_sigma', 0.0, 6.0, initial_Vpp_sigma, 0.1)
Vpp_pi = st.slider('Vpp_pi', -5.0, 0.0, initial_Vpp_pi, 0.1)

# Diccionarios de parámetros actualizados
sk_params = {
    "Vss_sigma": Vss_sigma,
    "Vsp_sigma": Vsp_sigma,
    "Vpp_sigma": Vpp_sigma,
    "Vpp_pi":   Vpp_pi
}
onsite_params = {
    "E_s_Ga": Es_Ga,
    "E_p_Ga": Ep_Ga,
    "E_s_As": Es_As,
    "E_p_As": Ep_As
}

# Construye el modelo con los parámetros actualizados
model, orb_names, orb_coords, lat = build_gaas_model(sk_params, onsite_params)

# Define la ruta k y resuelve
k_points = [[0.25, 0.75, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.25, 0.75, 0.5], [0.375, 0.75, 0.375]]
(k_vec, k_dist, k_node) = model.k_path(k_points, 200)
k_label = ["W", "L", r"$\Gamma$", "X", "W", "K"]
evals = model.solve_all(k_vec)

# Grafica con Matplotlib
fig, ax = plt.subplots(figsize=(8, 6)) # Ajusta el tamaño según necesites
for band in evals:
    ax.plot(k_dist, band)
ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
ax.set_ylabel("Energía (eV)")
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)
ax.set_xlim(k_node[0], k_node[-1])
ax.set_ylim(-9, 13)
for x in k_node:
    ax.axvline(x=x, color='k', linewidth=0.5)
ax.set_title("Bandas del GaAs (Modelo TB)")

# Muestra la figura en Streamlit
st.pyplot(fig)
using CUDA, Plots, DelimitedFiles, Trapz, LaTeXStrings



# ---------------------------------------------------------------------------- #
#                        Molecular Cross-Sections (GPU)                        #
# ---------------------------------------------------------------------------- #


function line_broadener!(spectrum::CuDeviceMatrix{Float32, 1}, indices::CuDeviceVector{Int32, 1}, x::CuDeviceVector{Float32, 1}, intensity::CuDeviceVector{Float32, 1}, 
    g_air::CuDeviceVector{Float32, 1}, g_self::CuDeviceVector{Float32, 1}, n_air::CuDeviceVector{Float32, 1}, molar_mass::Float32,
    T_ref::Float32, PT::CuDeviceMatrix{Float32, 1}, p_partial::Float32, c_mps::Float32, avogadro::Float32, kb::Float32)
    """
    Computes the Voigt profile using a sparse method, which only broadens non-zero spectral lines (given by the indices array)
    NOTE: Temporarily using a fixed window for the wings (this should be adaptive! It will be updated in the next version)
          Additionally, there is an odd bug that only appears in the GPU implementation, which affects the boundaries
            For windows that overlap with the boundary, the wings are miscalculated on GPU, even though the identical CPU 
            implementation below (literally a line-for-line copy) computes the boundary region correctly
            It only affects the first/last 400 grid points out of 319K (for H2O), so the impact on the results is negligible 
            but I've been discussing it with the CUDA.jl developers to try to track down and fix the cause 
    """

    index, stride = threadIdx().x, blockDim().x
    for idx = index:stride:length(indices)
        i = indices[idx]
        for k = 1:size(PT,1)
            sigma = (x[i] / c_mps) * sqrt( 2 * avogadro * kb * PT[k,2] * log(2) / molar_mass ) 
            gamma = ( g_air[i]*(PT[k,1]-p_partial*PT[k,1]) + g_self[i]*p_partial*PT[k,1]) * (T_ref/PT[k,2])^n_air[i]
            d = (gamma - sigma) /(sigma + gamma)
            sigma2 = (1.0692 * gamma + sqrt( 0.86639*gamma^2 + 4*sigma^2))/2
            cg = 0.32460 - 0.61825*d + 0.17681*d^2 + 0.12109*d^3
            cl = 0.68188 + 0.61293*d - 0.18384*d^2 - 0.11568*d^3

            for j in max(i-200,1):1:min(i+200,length(x))     
                result = cg*sqrt(log(2)/pi) * exp(-log(2) * ((x[j]-x[i])/sigma2)^2 ) / sigma2
                result += cl*sigma2 / (pi * ( (x[j]-x[i])^2 + sigma2^2))        
                spectrum[j,k] += result * intensity[i] 
            end
        end
    end
    return
end


function original_line_broadener!(spectrum::CuDeviceMatrix{Float32, 1}, x::CuDeviceVector{Float32, 1}, intensity::CuDeviceVector{Float32, 1}, 
    g_air::CuDeviceVector{Float32, 1}, g_self::CuDeviceVector{Float32, 1}, n_air::CuDeviceVector{Float32, 1}, molar_mass::Float32,
    T_ref::Float32, PT::CuDeviceMatrix{Float32, 1}, p_partial::Float32, c_mps::Float32, avogadro::Float32, kb::Float32)
    """
    The original (non-sparse) implementation of 
    """

    index, stride = threadIdx().x, blockDim().x
    for i = index:stride:length(x)
        for k = 1:size(PT,1)
            sigma = (x[i] / c_mps) * sqrt( 2 * avogadro * kb * PT[k,2] * log(2) / molar_mass ) 
            gamma = ( g_air[i]*(PT[k,1]-p_partial*PT[k,1]) + g_self[i]*p_partial*PT[k,1]) * (T_ref/PT[k,2])^n_air[i]
            d = (gamma - sigma) /(sigma + gamma)
            sigma2 = (1.0692 * gamma + sqrt( 0.86639*gamma^2 + 4*sigma^2))/2
            cg = 0.32460 - 0.61825*d + 0.17681*d^2 + 0.12109*d^3
            cl = 0.68188 + 0.61293*d - 0.18384*d^2 - 0.11568*d^3
            for j in max(i-100,1):min(i+100,length(x))
                result = cg*sqrt(log(2)/pi) * exp(-log(2) * ((x[j]-x[i])/sigma2)^2 ) / sigma2
                result += cl*sigma2 / (pi * ( (x[j]-x[i])^2 + sigma2^2))        
                spectrum[j,k] += result * intensity[i] 
            end
        end
    end
    return
end

function line_broadener_CPU!(spectrum, indices, x, intensity, g_air, g_self, n_air, 
    molar_mass, T_ref, PT, p_partial, c_mps, avogadro, kb)
    """
    Computes the Voigt profile on CPU, intended for comparison with the GPU method (and highlights the boundary bug on GPU)
    It's a line-for-line copy and should produce the same results, but (unlike the GPU version) it computes boundaries correctly
    """
    for idx = 1:length(indices)
        i = indices[idx]
        for k = 1:size(PT,1)
            sigma = (x[i] / c_mps) * sqrt( 2 * avogadro * kb * PT[k,2] * log(2) / molar_mass ) 
            gamma = ( g_air[i]*(PT[k,1]-p_partial*PT[k,1]) + g_self[i]*p_partial*PT[k,1]) * (T_ref/PT[k,2])^n_air[i]
            d = (gamma - sigma) /(sigma + gamma)
            sigma2 = (1.0692 * gamma + sqrt( 0.86639*gamma^2 + 4*sigma^2))/2
            cg = 0.32460 - 0.61825*d + 0.17681*d^2 + 0.12109*d^3
            cl = 0.68188 + 0.61293*d - 0.18384*d^2 - 0.11568*d^3

            for j in max(i-200,1):1:min(i+200,length(x))
                result = cg*sqrt(log(2)/pi) * exp(-log(2) * ((x[j]-x[i])/sigma2)^2 ) / sigma2
                result += cl*sigma2 / (pi * ( (x[j]-x[i])^2 + sigma2^2))        
                spectrum[j,k] += result * intensity[i] 
            end
        end
    end
    return
end

# ---------------------------------------------------------------------------- #
#                           Atmospheric Profile Setup                          #
# ---------------------------------------------------------------------------- #

function T_scaling(Ts, p, p_sur, R_cp, gamma)
    return Ts * (p/p_sur).^(gamma*R_cp)
end

function bb_f(u)
    return (u.^3) ./ ( exp.(u) .- 1 )
end

function blackbody(u, T, f, C)
    return C * f(u) * (T^4) * (0.01^1) 
end


R = 8.3145                     #  J / (mol*K)
g = 9.81                       #  m / s^2
sb = 5.67e-8                   # W / (K^4 * m^2) 
R_cp = 2. / 7
avogadro = 6.023e23       

T_star = 5772.             # Kelvin
T_star = 4000.
R_star = 6.9634e8          # meters
planet_distance = 1.496e11
albedo = 0.3 
T_irr = T_star * sqrt(R_star/planet_distance)
T_int = 288.


N_layers = 50
p_sur = 101000.

Z = LinRange(1, 50000, N_layers)
H = 8400.                  # meters
P = p_sur * exp.(-Z ./ H) 



T_strat = 216.7
lapse_gamma = 0.6655845642089844 
T = T_scaling(T_int, P, p_sur, R_cp, lapse_gamma)
T[T .< T_strat] .= T_strat
PT = Float32.([ P.*9.86923e-6   T ]) 



# ---------------------------------------------------------------------------- #
#                                Opacity Setup                                 #
# ---------------------------------------------------------------------------- #

c_mps = 2.998e8
avogadro = 6.022e23
kb = 1.380649e-23 


hitran = readdlm("C:/Users/Zeyad/Downloads/65123146.out", ',')
headers = hitran[1,:]
hitran = hitran[2:end,:]


HITRAN_scaling = 1e19

h2o = hitran[:,1] .== 1;
co2 = hitran[:,1] .== 7;
nu_species = hitran[:,4]
Sij = hitran[:,5] * HITRAN_scaling
n_air = hitran[:,9]
gamma_air = hitran[:,7]
gamma_self = hitran[:,8]

M_h2o = 18.01528   
M_co2 = 44.01
T_ref = 296.
p_ref = 1.
p_partial = 0.022

idx = h2o;
molar_mass = M_h2o;
nu_spectrum = Float64.(nu_species[idx])



# ---------------------------------------------------------------------------- #
#                              Radiative Transfer                              #
# ---------------------------------------------------------------------------- #


cos_a = 1.              # Angle cosine, called mu_* in Guillot (2010)
k = 1.380649e-23        # J/K   =  m s-1
h = 6.62607015e-34      # J/Hz  =  m2 kg s-1
c = 299792458           # m/s
g_cm = 100 * 9.81       # cm/s
C = pi * (k^4) / ( (c^2) * (h^3) )



vis_start = argmax(nu_spectrum .> 12500)
vis_end = argmax(nu_spectrum .> 25000)
th_start = 1
th_end = argmax(nu_spectrum .> 12500)


u_int = nu_spectrum .* ((h*c)/(T_int*k)) ./ 0.01
J_int = blackbody.(u_int, T_int, bb_f, C) 
J_th = trapz( nu_spectrum[th_start:th_end], J_int[th_start:th_end] )
u_irr = nu_spectrum .* ((h*c)/(T_irr*k)) ./ 0.01
J_irr = blackbody.(u_irr, T_star, bb_f, C) 
J_vis = trapz( nu_spectrum[vis_start:vis_end], J_int[vis_start:vis_end] )



# ---------------------------------------------------------------------------- #
#                                Deployment                                    #
# ---------------------------------------------------------------------------- #


nu_gpu = cu( Float32.(nu_spectrum) )
intensity = cu(Float32.(Sij[idx]) )
g_air = cu( Float32.(gamma_air[idx]) )
g_self = cu( Float32.(gamma_self[idx]) )
n_air_gpu = cu( Float32.(n_air[idx]) )
molar_mass = Float32(molar_mass)
T_ref, p_partial = Float32(T_ref), Float32(p_ref), Float32(p_partial)
c_mps, avogadro, kb = Float32(c_mps), Float32(avogadro), Float32(kb)

indices = vec(1:length(intensity))
indices = indices[intensity .> 1e-5]
indices_gpu = cu( Int32.(indices) )

#   OMITTED SECTION  
""" The code that performs the p-T profile calculation has significant issues in implementation,
    so I've temporarily removed it from this code while I try to fix them.

    The code below shows an example of calculating cross sections using the GPU broadener, and
    compares it to the CPU implementation. Due to limitations in Float32 accuracy (which is the
    standard type for GPU computing) the line intensities are scaled so that significant spectral
    features are ~1-10, so the results need scaled back down to HITRAN units
"""

# A comparison between the sparse and non-sparse implementations of the GPU broadener
PT_gpu = cu( PT )
spectrum_gpu = cu( Float32.(zeros(length(nu_spectrum), N_layers) ) )

@cuda threads=640 line_broadener!(spectrum_gpu, indices_gpu, nu_gpu, intensity, g_air, g_self, n_air_gpu, molar_mass, T_ref, PT_gpu, p_partial, c_mps, avogadro, kb)
synchronize() 
cross_sections = Array(Float64.(spectrum_gpu)) 


spectrum_gpu2 = cu( Float32.(zeros(length(nu_spectrum), N_layers) ) )
@cuda threads=640 original_line_broadener!(spectrum_gpu2, nu_gpu, intensity, g_air, g_self, n_air_gpu, molar_mass, T_ref, PT_gpu, p_partial, c_mps, avogadro, kb)
synchronize() 
cross_sections2 = Array(Float64.(spectrum_gpu)) 

nu_spectrum = Array(nu_gpu)
plot( nu_spectrum[400:1400], cross_sections[400:1400,1]./HITRAN_scaling, label="Sparse GPU Implementation", linewidth=2, xlabel="Wavenumber (cm^{-1})", ylabel= L"Intensity (cm^{-1} / [molecule \cdot cm^{-2}])", yscale=:log10 )
plot!( nu_spectrum[400:1400], cross_sections2[400:1400,1]./HITRAN_scaling, label="Non-Sparse GPU Implementation", linestyle=:dash, linewidth=4, yscale=:log10, color="red" )


# Optional: Comparison with the CPU implementation (which highlights the boundary bug that appears in CUDA.jl's adaptation)
nu_spectrum = Array(nu_gpu)
spectrum = Float32.(zeros(length(nu_species[idx]), size(PT,1))) 
intensity_CPU = Array(intensity)
line_broadener_CPU!(spectrum, indices, nu_spectrum, intensity_CPU, gamma_air[idx], gamma_self[idx], n_air[idx], molar_mass, T_ref, PT, p_partial, c_mps, avogadro, kb)
cpu_cross_sections = Array(Float64.(spectrum)) 

nu_spectrum = Array(nu_gpu)
plot( nu_spectrum[1:1000], cross_sections[1:1000,1], label="GPU", linewidth=2 )
plot!( nu_spectrum[1:1000], cpu_cross_sections[1:1000,1], label="CPU", linestyle=:dash, linewidth=1.5 )
plot!( [ nu_spectrum[200], nu_spectrum[200] ] , [0, 3], linestyle=:dash, label="Index 200" )
plot!( [ nu_spectrum[400], nu_spectrum[400] ] , [0, 3], linestyle=:dash, label="Index 400" )

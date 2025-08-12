import DPFEHM
import PyPlot
using Statistics
using SparseArrays
using Pkg 
using Random: randperm
using LinearAlgebra
using PyPlot: subplots

# ADD YOUR OWN PATH TO fast-kappa.jl
include("fast-kappa.jl")
import .Quick_Kappa: maxsv as qk_maxsv, minsv as qk_minsv, kcond as qk_kcond, kprecond as qk_precond

function addfractures_xy!(logks, z_idx, fracture_logk, fracture_scales; beta=0.5, pattern=:default, z_thickness = 20)
    if z_thickness == 0
        z_thickness=1
    end
    if fracture_scales > 0
        if fracture_logk < 0.0
            @show fracture_logk
            @show fracture_scales
            error("you need to increase fracture_logk so that the permeability of the small fractures is greater than the permeability of the matrix")
        end
        
        n = size(logks, 1)
        nz = size(logks, 3)
        half_thick = fld(z_thickness, 2)
        z_start = max(1, z_idx - half_thick)
        z_end = min(nz, z_idx + half_thick)    

        if pattern == :default
            logks[div(n, 2), 1:div(3 * n, 4) + 1, z_start:z_end] .= fracture_logk + log(1.5 ^ beta)
            logks[div(n, 4):div(3 * n, 4), div(n, 2), z_start:z_end] .= fracture_logk
            addfractures_xy!(view(logks, 1:div(n, 2), div(n, 2) + 1:n, :), z_idx, fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta, pattern=pattern, z_thickness=div(z_thickness, 2))
            addfractures_xy!(view(logks, div(n, 2) + 1:n, div(n, 2) + 1:n, :), z_idx, fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta, pattern=pattern, z_thickness=div(z_thickness, 2))

        elseif pattern == :rotated_180
            logks[div(n, 2), div(n, 4):n, z_start:z_end] .= fracture_logk + log(1.5 ^ beta)
            logks[div(n, 4):div(3 * n, 4), div(n, 2)+1, z_start:z_end] .= fracture_logk
            
            addfractures_xy!(view(logks, 1:div(n, 2), 1:div(n, 2), :), z_idx, fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta, pattern=pattern, z_thickness=div(z_thickness, 2))
            addfractures_xy!(view(logks, div(n, 2) + 1:n, 1:div(n, 2), :), z_idx, fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta, pattern=pattern, z_thickness=div(z_thickness, 2))

        else
            error("Unknown pattern: $pattern. Valid options are :default, :reflected, :rotated_90, :rotated_180, :rotated_270")
        end
    end
end






function addboundaries_3d(logks)
    nx, ny, nz = size(logks)
    logks_x = cat(logks[1:1, :, :], logks, logks[end:end, :, :], dims=1)
    logks_xy = cat(logks_x[:, 1:1, :], logks_x, logks_x[:, end:end, :], dims=2)
    logks_xyz = cat(logks_xy[:, :, 1:1], logks_xy, logks_xy[:, :, end:end], dims=3)
    
    return logks_xyz
end

function fractal_fractures_3d(N, fracture_scales; doplot=false, matrix_logk=0.0, fracture_logk=1.0, kwargs...)
    mins = [0.0, 0.0, 0.0]
    maxs = [1.0, 1.0, 1.0]
    ns = [N + 2, N + 2, N + 2]
    xs = range(mins[1], maxs[1]; length=ns[1])
    ys = range(mins[2], maxs[2]; length=ns[2])
    zs = range(mins[3], maxs[3]; length=ns[3])
    
    logks = fill(matrix_logk, N, N, N)

    center_idx = div(N, 2)

    addfractures_xy!(logks, center_idx, fracture_logk, fracture_scales; pattern=:default, z_thickness = div(N,2))
    addfractures_xy!(logks, center_idx, fracture_logk, fracture_scales; pattern=:rotated_180, z_thickness = div(N,2))
    logks = addboundaries_3d(logks)
    
    if doplot
        fig, axs = PyPlot.subplots(2, 2, figsize=(12, 10))
        
        im1 = axs[1,1].imshow(logks[:, :, div(end, 2)], origin="lower")
        axs[1,1].set_title("XY slice (center Z)")
        fig.colorbar(im1, ax=axs[1,1])
        
        xz_avg_proj = mean(logks, dims=2)[:, 1, :]
        im2 = axs[1,2].imshow(xz_avg_proj', origin="lower", aspect="auto", alpha = 0.95)
        axs[1,2].set_title("XZ slice (average across Y)")
        
        yz_max_proj = mean(logks, dims=1)[1, :, :]
        im3 = axs[2,1].imshow(yz_max_proj', origin="lower", aspect="auto")
        axs[2,1].set_title("YZ slice (average across X)")
        axs[2,1].set_xlabel("Y direction")
        axs[2,1].set_ylabel("Z direction")
        
        axs[2,2].remove()
        ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
        
        high_perm_indices = findall(x -> x > matrix_logk + 0.1, logks)
        indec = CartesianIndices(logks)
        if !isempty(high_perm_indices)
            x_coords = [i[1] for i in high_perm_indices]
            y_coords = [i[2] for i in high_perm_indices]
            z_coords = [i[3] for i in high_perm_indices]
            colors = [logks[i] for i in high_perm_indices]
            
            scatter = ax_3d.scatter(x_coords, y_coords, z_coords, c=colors, alpha=0.6, s=10)
        end
        ax_3d.set_title("3D Fracture Network")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        
        PyPlot.tight_layout()
        display(fig)
        PyPlot.savefig("3D_pitchfork.pdf")
        PyPlot.close(fig)
    end
    
    coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid3d(mins, maxs, ns)

    logKs2Ks_neighbors(Ks) = exp.(0.5 .* (Ks[map(p -> p[1], neighbors)] .+ Ks[map(p -> p[2], neighbors)]))
    Ks_neighbors = logKs2Ks_neighbors(vec(logks))

    nx, ny, nz = ns
    node_idx = reshape(1:(nx * ny * nz), nx, ny, nz)
    interior_nodes = vec(node_idx[2:end-1, 2:end-1, 2:end-1])  # these are the N^3 free nodes

    all_nodes = collect(1:(nx * ny * nz))
    boundary_nodes = setdiff(all_nodes, interior_nodes)
    dirichleths = zeros(nx * ny * nz)
    Qs = zeros(nx * ny * nz)

    bottom_nodes = findall(i -> coords[3, i] ≈ mins[3], 1:size(coords, 2))
    top_nodes    = findall(i -> coords[3, i] ≈ maxs[3], 1:size(coords, 2))
    dirichletnodes = union(bottom_nodes, top_nodes)
    dirichleths[bottom_nodes] .= 1.0
    dirichleths[top_nodes] .= 0.0

    A_full = groundwater_h_full(nx * ny * nz, Ks_neighbors, neighbors, areasoverlengths)
    b_full = copy(Qs)

    for node in dirichletnodes
        A_full[node, :] .= 0.0
        A_full[:, node] .= 0.0
        A_full[node, node] = 1.0
        b_full[node] = dirichleths[node]
    end

    A = A_full[interior_nodes, interior_nodes]
    b = b_full[interior_nodes]
    x = A \ b
    h = zeros(nx * ny * nz)
    h[interior_nodes] = x
    h[dirichletnodes] = dirichleths[dirichletnodes]  # impose known values

    return A, x, b, h, logks
end
function groundwater_h_full(n_nodes, Ks_neighbors, neighbors, areasoverlengths)
    A = spzeros(n_nodes, n_nodes)
    for (i, ((n1, n2), k, aol)) in enumerate(zip(neighbors, Ks_neighbors, areasoverlengths))
        A[n1, n1] += k * aol
        A[n2, n2] += k * aol
        A[n1, n2] -= k * aol
        A[n2, n1] -= k * aol
    end
    return A
end
function print_matrix_pretty(M)
    open("matrix_pretty", "w") do io
        for row in eachrow(M)
            println(io, join(rpad(string(round(x, digits=6)), 10) for x in row))
        end
    end
end

function random_b(N::Int, num_sites::Int)
    b = zeros(N)
    idxs = randperm(N)[1:num_sites]
    signs = rand([-1, 1], num_sites)
    b[idxs] = signs
    return b
end

function log_inv_eps(A, b)
    x = A \ b
    x ./= norm(x)
    nonzeros = abs.(x[abs.(x) .> 1e-12])
    return log(1 / minimum(nonzeros))
end

function run_experiment()
    powers = 2:6          
    Ns = [2^i for i in powers]
    x_vals = [3i for i in powers]
    fracture_scales = 0
    num_trials = 100
    num_sites = 20

    mean_logs = Float64[]
    std_logs = Float64[]

    for N in Ns
        println("Running N = $N (matrix size = $(N^3) × $(N^3))")
        logs = Float64[]
        try
            A, _, _, _, _ = fractal_fractures_3d(N, fracture_scales; doplot=false)
            for _ in 1:num_trials
                b = random_b(size(A, 1), num_sites)
                push!(logs, log_inv_eps(A, b))
            end
            push!(mean_logs, mean(logs))
            push!(std_logs, std(logs))
        catch e
            @warn "Failed at N = $N: $e"
            push!(mean_logs, NaN)
            push!(std_logs, NaN)
        end
    end

    # Plot
    _, ax = subplots()
    ax.errorbar(x_vals, mean_logs, yerr=std_logs, fmt="o-", capsize=4)
    xticks_to_show = collect(range(minimum(x_vals), maximum(x_vals); step=2))
    ax.set_xticks(xticks_to_show)
    ax.set_xlabel("log(N)")
    ax.set_ylabel("log(1/ε)")
    ax.set_title("Scaling of logrithmic error with system size")
    ax.grid(true)
    PyPlot.tight_layout()
    PyPlot.savefig("scaling-x.pdf")
    println("Saved plot to scaling-x.pdf")
end



function count_distinct_entries(A)
    return length(unique(A.nzval))
end
using PyPlot
using Printf
function plot_unique_vlas_vs_fracture_scale(max_i)
	bs = 0:max_i-1  # fracture scales
	max_distinct_counts = Int[]  # store max distinct element count for each b

	for b in bs
		i_vals = (b + 1):max_i
		distinct_counts = Int[]
		
		for i in i_vals 

			N = 2 ^ i
			A, _, _, _, _ = fractal_fractures_3d(N, b; fracture_logk=5.0, doplot=false)
			ndistinct = count_distinct_entries(A)
			push!(distinct_counts, ndistinct)
		end
		
		push!(max_distinct_counts, maximum(distinct_counts))  # store max for this b
	end


	x = collect(bs)
	y = max_distinct_counts

	a = cov(x, y) / var(x)
	c = mean(y) - a * mean(x)
	y_fit = a .* x .+ c


	figure()
	plot(x, y, marker="o", linestyle="-", label="Data")
	plot(x, y_fit, linestyle="--", color="red", label="Best fit line")
	xlabel("Number of Fracture Scales")
	ylabel("Max Distinct Elements in A")
	title("Max Distinct Elements vs. Fracture Scales")
	legend()
	tight_layout()
	eq_str = @sprintf("y = %.3f x + %.3f", a, c)

	text(x[1] + 0.1, maximum(y) * 0.9, eq_str, fontsize=12, color="red", bbox=Dict("facecolor"=>"white", "alpha"=>0.5, "pad"=>5))
	display(gcf())
	savefig("MaxDistinct_vs_FractureScale_Pitchfork.pdf")
end
function plot_condition_numbers(t::Int)
    Ns = Int[]
    conds = Float64[]
    ref_line = Float64[]

    for i in 1:t
        println("i = ", i)
        N = 2^i
        push!(Ns, 3*i)

        A, x, b, h, logks = fractal_fractures_3d(N, i-1; doplot=false, matrix_logk=0.0, fracture_logk=2.0)
        println("created A")
        κ = qk_kcond(A)
        push!(conds, log2(κ))

        push!(ref_line, 2*i)
    end

    Ns_f = Float64.(Ns)

    # Plot
    figure()
    plot(Ns_f, conds, "o-", label="effective condition number of G' ")
    plot(Ns_f, ref_line, "--", label="N^(2/3)")

    xlabel("Dimension log(N)")
    ylabel("log(Condition Number)")
    title("Condition Number vs N")
    legend()
    tight_layout()
    savefig("cond_num_scaling.pdf")
end

; 
module JuSW_ESPH

    using Revise
    using Base.Threads
    using Printf
    using CUDA

    include("parameters.jl")
    include("build_grid.jl")
    include("link_list.jl")
    include("kernel.jl")
    include("bottom_friction_interpolation.jl")
    include("backendtools.jl")
    using .BackendTools
    include("apply_Friction.jl")
    include("update_particles.jl")
    include("update_openboundary.jl")
    include("approx_RiemannSolver.jl")
    include("apply_fluxes.jl")

    export JuSW_ESPH_simulation!, save_particle_data

    function JuSW_ESPH_simulation!(x, y, wse, h, vx, vy, hvx, hvy, z_b, fr_manning, volume, itype, hsml, MAX_NB_FLUID, MAX_NB_BOTTOM, int_fluid, int_openboundary, int_virtual, int_bottom, ntotal, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, IDbc, numOpenBound, numData_OpenBound, timeOpenBound, 
                                    h_timeOpenBound, vx_timeOpenBound, fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, Openboundary_neighbor, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, 
                                    Openboundary_dWijdY, bottom_neighbor, bottom_dxij, bottom_dyij, bottom_Rij, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, 
                                    Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy)

        numthreads = 32   # 1024 maximum
        numblocks = cld(length(int_fluid), numthreads)

        # Nearest Neighboring Particles Search (NNPS) - Link-List method
        println("Searching Nearest Neighboring Particles - Link-List method - ....")
        GridParams = build_grid(x, y, hsml, skf, ntotal)
        link_list!(GridParams, x, y, hsml, itype, MAX_NB_FLUID, MAX_NB_BOTTOM, int_openboundary, int_virtual, ntotal, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, fluid_neighbor, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, fluid_dxij, fluid_dyij, fluid_Rij, 
        Openboundary_neighbor, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, bottom_neighbor, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, bottom_dxij, bottom_dyij, bottom_Rij)

        # Move data to the selected backend (GPU here) for computation
        backend = default_backend()
        println("Running on ", backend)
        println("Converting arrays to CUDA arrays ...")
        x, y, wse, h, vx, vy, hvx, hvy, z_b, fr_manning, volume, itype, hsml, int_fluid, int_openboundary, int_virtual, int_bottom, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, IDbc, numData_OpenBound, timeOpenBound, h_timeOpenBound, vx_timeOpenBound, 
        fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, Openboundary_neighbor, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, 
        h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, 
        BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy, bottom_neighbor, bottom_Wij = 
        to_backend(backend, (x, y, wse, h, vx, vy, hvx, hvy, z_b, fr_manning, volume, itype, hsml, int_fluid, int_openboundary, int_virtual, int_bottom, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, IDbc, numData_OpenBound, timeOpenBound, h_timeOpenBound, 
        vx_timeOpenBound, fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, Openboundary_neighbor, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, 
        h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, 
        BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy, bottom_neighbor, bottom_Wij))

        # Bottom and friction interpolation to fluid particles
        println("Interpolating Bottom and Friction to particles ....")
        @cuda threads=numthreads blocks=numblocks bottom_friction_interpolation_kernel!(int_fluid, h, hvx, hvy, wse, vx, vy, z_b, fr_manning, volume, MAX_NB_BOTTOM, bottom_neighbor, bottom_Wij)

        # Time integration parameters
        time = 0.0                                      # Start simulation time
        n_dt = 0                                        # Number of time steps
        Print = 0.0                                     # Print frequency
        dt = dt_max                                     # Time step size
                
        while time <= t_end # loop over time step
            
            # display progress
            prog = ceil((100*time)/t_end)    
            println("progress = $prog%,  time = $time,  timestep = $n_dt,   dt = $dt")
            
            n_dt = n_dt + 1
            
            if n_dt != 1
                h_o[int_fluid]      .= h[int_fluid]
                hvx_o[int_fluid] .= hvx[int_fluid]
                hvy_o[int_fluid] .= hvy[int_fluid]
            
                @cuda threads=numthreads blocks=numblocks apply_Friction_kernel!(int_fluid, vx, vy, h, hvx, hvy, fr_manning, d_hvx_dt, d_hvy_dt, grav)
                @cuda threads=numthreads blocks=numblocks update_particles_half_step_kernel!(int_fluid, h, wse, z_b, vx, vy, hvx, hvy, d_h_dt, d_hvx_dt, d_hvy_dt, dt)
            end

            if !isempty(int_openboundary)
                update_openboundary!(int_openboundary, wse, h, vx, vy, z_b, volume, itype, MAX_NB_FLUID, IDbc, numOpenBound, numData_OpenBound, timeOpenBound, h_timeOpenBound, vx_timeOpenBound, time, boundary_normals_X, boundary_normals_Y, Openboundary_neighbor, Openboundary_Wij, numthreads)
            end

            @cuda threads=numthreads blocks=numblocks approx_RiemannSolver_kernel!(int_fluid, h, wse, z_b, vx, vy, itype, MAX_NB_FLUID, fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, wall_normals_X, wall_normals_Y, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, 
                                  Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy, grav)

            @cuda threads=numthreads blocks=numblocks apply_fluxes_kernel!(int_fluid, h, volume, itype, MAX_NB_FLUID, fluid_neighbor, fluid_dWijdR, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_hvx, BedSource_Left_hvy, d_h_dt, d_hvx_dt, d_hvy_dt)

           # determine time step size
            hsml_min = minimum(hsml[int_fluid])
            vel_max  = maximum((sqrt.(vx[int_fluid].^2 .+ vy[int_fluid].^2)))
            c_max    = maximum(sqrt.(grav .* h[int_fluid]))

            dt = CFL * (hsml_min / (vel_max + c_max))
        
            if (Print + dt) > print_step
                dt = print_step - Print
            end
                        
            if n_dt == 1
                @cuda threads=numthreads blocks=numblocks update_particles_half_step_kernel!(int_fluid, h, wse, z_b, vx, vy, hvx, hvy, d_h_dt, d_hvx_dt, d_hvy_dt, dt)
            else
                @cuda threads=numthreads blocks=numblocks update_particles_full_step_kernel!(int_fluid, h, h_o, wse, z_b, vx, vy, hvx, hvy, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, dt)              
            end

            # ------------------------------------
            # Save particle data every print step
            # ------------------------------------
            Print = Print + dt
            if isapprox(Print, print_step, atol=1e-14)
                
                x_cpu   = Array(x)
                y_cpu   = Array(y)
                vx_cpu  = Array(vx)
                vy_cpu  = Array(vy)
                wse_cpu = Array(wse)
                h_cpu   = Array(h)
                z_b_cpu = Array(z_b)
                itype_cpu = Array(itype)
                int_fluid_cpu = Array(int_fluid)
                hsml_cpu = Array(hsml)

                # Save data to file
                fname = "benchmarks/$(case)/Data_at_time_$(round(time,digits=2)).dat"
                save_particle_data(x_cpu, y_cpu, vx_cpu, vy_cpu, wse_cpu, h_cpu, z_b_cpu, itype_cpu, hsml_cpu, int_fluid_cpu, fname)
                Print = 0.

                x   = CuArray(x_cpu)
                y   = CuArray(y_cpu)
                vx  = CuArray(vx_cpu)
                vy  = CuArray(vy_cpu)
                wse = CuArray(wse_cpu)
                h  = CuArray(h_cpu)
                z_b = CuArray(z_b_cpu)
                itype = CuArray(itype_cpu)
                int_fluid = CuArray(int_fluid_cpu)
                hsml = CuArray(hsml_cpu)
                
            end

            # Update  time
            time  = time + dt
            
        end
        
        return nothing
    end

    function save_particle_data(x_cpu, y_cpu, vx_cpu, vy_cpu, wse_cpu, h_cpu, z_b_cpu, itype_cpu, hsml_cpu, int_fluid_cpu, fname)
        
        # Find centerline particles
        y_2 = maximum(y_cpu[:]) / 2
        centerline_y_indices = zeros(Int64, length(int_fluid_cpu))
        ii = 0
        for i in int_fluid_cpu
            if isapprox(y_cpu[i], y_2, atol = hsml_cpu[i]/2)
                ii += 1
                centerline_y_indices[ii] = i
            end
        end

        # Save all particle data
        open(fname, "w") do f
            for i in int_fluid_cpu
                @printf(f, "%d %f %f %f %f %f %f %f %d\n",
                    i, x_cpu[i], y_cpu[i], vx_cpu[i], vy_cpu[i], wse_cpu[i], h_cpu[i], z_b_cpu[i], itype_cpu[i])
            end
        end

        # Save centerline particles if any
        centerline_file = replace(fname, ".dat" => "_centerline.dat")
        ii = count(!iszero, centerline_y_indices)
        if ii > 0
            open(centerline_file, "w") do f
                for j in 1:ii
                    i = centerline_y_indices[j]
                    @printf(f, "%d %f %f %f %f %f %f %f %d\n",
                        i, x_cpu[i], y_cpu[i], vx_cpu[i], vy_cpu[i], wse_cpu[i], h_cpu[i], z_b_cpu[i], itype_cpu[i])
                end
            end
        end

        return nothing
    end

end
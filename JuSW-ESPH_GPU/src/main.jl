# Read data from files
println("Reading data from files ....")
x, y, vx, vy, wse, z_b, fr_manning, volume, hsml, itype, IDbc, numOpenBound, numData_OpenBound, timeOpenBound, h_timeOpenBound, vx_timeOpenBound, int_fluid, int_openboundary, int_virtual, int_bottom, num_fluid, num_openbc, num_virtual, num_bottom, ntotal = read_data()

# Arrays allocation
const MAX_NB_FLUID  = 8
const MAX_NB_BOTTOM = 16
(fluid_neighbor, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, fluid_dxij, fluid_dyij, fluid_Rij, Openboundary_neighbor, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, 
bottom_neighbor, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, bottom_dxij, bottom_dyij, bottom_Rij, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, h, hvx, hvy, h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, Flux_LeftState_h, Flux_LeftState_hvx, 
Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy) = 
preallocate_Arrays(ntotal, int_fluid, int_openboundary, MAX_NB_FLUID, MAX_NB_BOTTOM)

# Simulation loop
println("Starting simulation ... \n")
@time begin
    JuSW_ESPH_simulation!(x, y, wse, h, vx, vy, hvx, hvy, z_b, fr_manning, volume, itype, hsml, MAX_NB_FLUID, MAX_NB_BOTTOM, int_fluid, int_openboundary, int_virtual, int_bottom, ntotal, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, IDbc, numOpenBound, numData_OpenBound, timeOpenBound, h_timeOpenBound, vx_timeOpenBound, 
                                                        fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, Openboundary_neighbor, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, bottom_neighbor, bottom_dxij, 
                                                        bottom_dyij, bottom_Rij, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, 
                                                        Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy)
end
println("progress = 100%, Simulation completed \n")

# Plot results
println("Ploting results ....")
visualize(x, y, int_fluid)
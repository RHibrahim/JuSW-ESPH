function preallocate_Arrays(ntotal, int_fluid, int_openboundary, MAX_NB_FLUID, MAX_NB_BOTTOM)

    fluid_neighbor      = zeros(Int, MAX_NB_FLUID, length(int_fluid))
    fluid_Wij           = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_dWijdR        = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_dWijdX        = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_dWijdY        = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_dxij          = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_dyij          = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    fluid_Rij           = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    Openboundary_neighbor = zeros(Int, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_Wij      = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_dWijdR   = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_dWijdX   = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_dWijdY   = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_dxij     = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_dyij     = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    Openboundary_Rij      = zeros(Float64, MAX_NB_FLUID, length(int_fluid) + length(int_openboundary))
    
    bottom_neighbor     = zeros(Int, MAX_NB_BOTTOM, ntotal)
    bottom_Wij          = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_dWijdR       = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_dWijdX       = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_dWijdY       = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_dxij         = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_dyij         = zeros(Float64, MAX_NB_BOTTOM, ntotal)
    bottom_Rij          = zeros(Float64, MAX_NB_BOTTOM, ntotal)

    wall_normals_X     = zeros(Float64, ntotal, 2)
    wall_normals_Y     = zeros(Float64, ntotal, 2)
    boundary_normals_X = zeros(Float64, ntotal, 2)
    boundary_normals_Y = zeros(Float64, ntotal, 2)

    h       = zeros(Float64, ntotal)
    hvx     = zeros(Float64, length(int_fluid))
    hvy     = zeros(Float64, length(int_fluid))
    h_o     = zeros(Float64, length(int_fluid))
    hvx_o   = zeros(Float64, length(int_fluid))
    hvy_o   = zeros(Float64, length(int_fluid))
    d_h_dt  = zeros(Float64, length(int_fluid))
    d_hvx_dt = zeros(Float64, length(int_fluid))
    d_hvy_dt = zeros(Float64, length(int_fluid))
	
	Flux_LeftState_h   = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_LeftState_hvx = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_LeftState_hvy = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    Flux_RightState_h   = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_RightState_hvx = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_RightState_hvy = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    Flux_Interface_h   = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_Interface_hvx = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    Flux_Interface_hvy = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    
    BedSource_Left_h   = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    BedSource_Left_hvx = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    BedSource_Left_hvy = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    BedSource_Right_h   = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    BedSource_Right_hvx = zeros(Float64, MAX_NB_FLUID, length(int_fluid))
    BedSource_Right_hvy = zeros(Float64, MAX_NB_FLUID, length(int_fluid))

    return (fluid_neighbor, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, fluid_dxij, fluid_dyij, fluid_Rij, Openboundary_neighbor, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, bottom_neighbor, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, bottom_dxij, bottom_dyij, bottom_Rij, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, h, hvx, hvy, h_o, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt,
            Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy)
end
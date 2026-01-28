function read_data()

    #------------- Read All data -------------#

    # read fluid data
    ID = 0
    int_fluid = Int64[]
    int_openboundary = Int64[]
    int_virtual = Int64[]
    int_bottom = Int64[]
    x = Float64[]
    y = Float64[]
    vx = Float64[]
    vy = Float64[]
    wse = Float64[]
    volume = Float64[]
    hsml = Float64[]
    itype = Int64[]
    z_b = Float64[]
    fr_manning = Float64[]
    
    ibc = Int64[]
    inflow_time = Float64[]
    inflow_h = Float64[]
    inflow_vx = Float64[]
    outflow_time = Float64[]
    outflow_h = Float64[]   
    outflow_vx = Float64[]

    filename = "benchmarks/$(case)/initial_data/Fluid_part.dat"
    open(filename, "r") do f
        for line in eachline(f)
            isempty(strip(line)) && continue
            startswith(line,"#") && continue

            v = split(line)
            ID = ID + 1
            push!(int_fluid, ID)
            push!(x, parse(Float64,v[2]))
            push!(y, parse(Float64,v[3]))
            push!(vx,parse(Float64,v[4]))
            push!(vy,parse(Float64,v[5]))
            push!(wse,parse(Float64,v[6]))
            push!(volume,parse(Float64,v[7]))
            push!(hsml,parse(Float64,v[8]))
            push!(itype,parse(Int64,v[9]))
            push!(z_b, 0.0)
            push!(fr_manning, 0.0)

        end
    end
    
    num_fluid = length(int_fluid)
    println("Number of fluid particles read: ", num_fluid)

    # end read fluid data

    # read open boundary data
    numOpenBound = 0
    filename = "benchmarks/$(case)/initial_data/Open_boundary_part.dat"

    if isfile(filename)

        open(filename, "r") do f
            # read open boundary data
            for line in eachline(f)
                isempty(strip(line)) && continue
                startswith(line,"#") && continue

                v = split(line)
                ID = ID + 1
                push!(int_openboundary, ID)
                push!(x, parse(Float64,v[2]))
                push!(y, parse(Float64,v[3]))
                push!(vx, 0.0)
                push!(vy, 0.0)
                push!(wse, 0.0)
                push!(volume,parse(Float64,v[4]))
                push!(itype,parse(Int64,v[5]))
                push!(hsml,parse(Float64,v[6]))
                push!(ibc,parse(Int64,v[7]))
                push!(z_b, 0.0)
                push!(fr_manning, 0.0)
            end
        end

        # Read open boundary time-dependent data
        fname_inflow = "benchmarks/$(case)/initial_data/Inflow.dat"
        
        if isfile(fname_inflow)
            numOpenBound = numOpenBound + 1
            open(fname_inflow, "r") do f
                for line in eachline(f)
                    isempty(strip(line)) && continue
                    startswith(line,"#") && continue

                    fields = split(line)

                    push!(inflow_time, parse(Float64,fields[1]))
                    push!(inflow_h, parse(Float64,fields[2]))
                    push!(inflow_vx, parse(Float64,fields[3]))

                    println("Inflow data: time=$(inflow_time), h=$(inflow_h), vx=$(inflow_vx)")
                end
            end
        else
            @info "No inflow data file found: $fname"
        end

        fname_outflow = "benchmarks/$(case)/initial_data/Outflow.dat"
        
        if isfile(fname_outflow)
            numOpenBound = numOpenBound + 1
            open(fname_outflow, "r") do f
                for line in eachline(f)
                    isempty(strip(line)) && continue
                    startswith(line,"#") && continue

                    fields = split(line)
                    
                    push!(outflow_time, parse(Float64,fields[1]))
                    push!(outflow_h, parse(Float64,fields[2]))
                    push!(outflow_vx, parse(Float64,fields[3]))

                    println("Outflow data: time=$(outflow_time), h=$(outflow_h), vx=$(outflow_vx)")
                end
            end
        else
            @info "No Outflow data file found: $fname"
        end
        
        numData = max(length(inflow_time), length(outflow_time))
        numData_OpenBound = zeros(Int64, numOpenBound)
        timeOpenBound = zeros(Float64, numData, numOpenBound)
        h_timeOpenBound = zeros(Float64, numData, numOpenBound)
        vx_timeOpenBound = zeros(Float64, numData, numOpenBound)

        if isfile(fname_inflow) # only inflow
            numData_OpenBound[1] = length(inflow_time)
            for i in 1:length(inflow_time)
                timeOpenBound[i, 1] = inflow_time[i]
                h_timeOpenBound[i, 1] = inflow_h[i]
                vx_timeOpenBound[i, 1] = inflow_vx[i]
            end
            if isfile(fname_outflow) # both inflow and outflow
                numData_OpenBound[2] = length(outflow_time)
                for i in 1:length(outflow_time)
                    timeOpenBound[i, 2] = outflow_time[i]
                    h_timeOpenBound[i, 2] = outflow_h[i]
                    vx_timeOpenBound[i, 2] = outflow_vx[i]
                end
            end
        elseif isfile(fname_outflow) # only outflow
            numData_OpenBound[1] = length(outflow_time)
            for i in 1:length(outflow_time)
                timeOpenBound[i, 1] = outflow_time[i]
                h_timeOpenBound[i, 1] = outflow_h[i]
                vx_timeOpenBound[i, 1] = outflow_vx[i]
            end
        end
    else
        @info "No open boundary file found: $filename"
        numData_OpenBound = zeros(Int64, 1)
        timeOpenBound = zeros(Float64, 1, 1)
        h_timeOpenBound = zeros(Float64, 1, 1)
        vx_timeOpenBound = zeros(Float64, 1, 1)
    end

    num_openboundary = length(int_openboundary)
    println("Number of open boundary particles read: ", num_openboundary)

    IDbc = zeros(Int64, num_openboundary + num_fluid)
    for i in 1:num_openboundary
        IDbc[i+num_fluid] = ibc[i]
    end

    # end read open boundary data

    # read virtual data
    filename = "benchmarks/$(case)/initial_data/Virtual_part.dat"
    
    if isfile(filename)
        open(filename, "r") do f
            for line in eachline(f)
                isempty(strip(line)) && continue
                startswith(line,"#") && continue
    
                v = split(line)
                ID = ID + 1
                push!(int_virtual, ID)
                push!(x, parse(Float64,v[2]))
                push!(y, parse(Float64,v[3]))
                push!(vx, 0.0)
                push!(vy, 0.0)
                push!(wse, 0.0)
                push!(volume,parse(Float64,v[6]))
                push!(itype,parse(Int64,v[7]))
                push!(hsml,parse(Float64,v[8]))
                push!(z_b, 0.0)
                push!(fr_manning, 0.0)
            end
        end
    else
        @info "No virtual particle file found: $filename"
    end
    
    num_virtual = length(int_virtual)
    println("Number of virtual particles read: ", num_virtual)

    # end read virtual data

    # read bottom data
    filename = "benchmarks/$(case)/initial_data/Bottom_part.dat"
    
    if isfile(filename)
        open(filename, "r") do f
            for line in eachline(f)
                isempty(strip(line)) && continue
                startswith(line,"#") && continue
    
                v = split(line)
                ID = ID + 1
                push!(int_bottom, ID)
                push!(x, parse(Float64,v[2]))
                push!(y, parse(Float64,v[3]))
                push!(vx, 0.0)
                push!(vy, 0.0)
                push!(wse, 0.0)
                push!(z_b,parse(Float64,v[4]))
                push!(fr_manning,parse(Float64,v[5]))
                push!(hsml, parse(Float64,v[6]))
                push!(volume,parse(Float64,v[7]))
                push!(itype,parse(Int64,v[8]))
            end
        end
    else
        @info "No bottom particle file found: $filename"
    end

    num_bottom = length(int_bottom)
    println("Number of bottom particles read: ", num_bottom)

    # end read bottom data

    ntotal = length(x)
    println("Total number of particles read: ", ntotal)
    #------------- End read data -------------#

    return x, y, vx, vy, wse, z_b, fr_manning, volume, hsml, itype, IDbc, numOpenBound, numData_OpenBound, timeOpenBound, h_timeOpenBound, vx_timeOpenBound, int_fluid, int_openboundary, int_virtual, int_bottom, num_fluid, num_openboundary, num_virtual, num_bottom, ntotal

end
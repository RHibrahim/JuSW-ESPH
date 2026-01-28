function kernel!(i, Rij, dxij, dyij, hsmlij, count, Wij, dWijdx, dWijdy, dWijdR)

    q = Rij / hsmlij

    if skf == 1
        
        factor = 15.0 / (7.0 * π * hsmlij^2)
        
        if 0 <= q <= 1.0
            
            Wij[count,i]    = factor * (2/3 - q^2 + 0.5 * q^3)
            dWijdR[count,i] = factor * (-2 * q + 3/2 * q^2) / hsmlij
            dWijdx[count,i] = factor * ((-2 * q + 3/2 * q^2) / hsmlij) * (dxij / Rij)
            dWijdy[count,i] = factor * ((-2 * q + 3/2 * q^2) / hsmlij) * (dyij / Rij)
            
        elseif 1.0 < q <= 2
            
            Wij[count,i]    = factor * 1/6 * (2 - q)^3
            dWijdR[count,i] = -factor * 1/6 * 3 * (2 - q)^2 / hsmlij
            dWijdx[count,i] = -factor * (1/6 * 3 * (2 - q)^2 / hsmlij) * (dxij / Rij)
            dWijdy[count,i] = -factor * (1/6 * 3 * (2 - q)^2 / hsmlij) * (dyij / Rij)
        
        end

    elseif skf == 2
        
        factor = 1.0 / (hsmlij^dim * π^(dim/2))
        
        if 0 <= q <= 3
            
            Wij[count,i]    = factor * exp(-q^2)
            dWijdR[count,i] = Wij[count,i] * (-2 * q / hsmlij)
            dWijdx[count,i] = Wij[count,i] * (-2 * dxij / hsmlij^2)
            dWijdy[count,i] = Wij[count,i] * (-2 * dyij / hsmlij^2)
            
        end

    elseif skf == 3
        
        factor = 7.0 / (478.0 * π * hsmlij^2)

        if 0 <= q <= 1

            Wij[count,i]    = factor * ((3 - q)^5 - 6 * (2 - q)^5 + 15 * (1 - q)^5)
            dWijdr[count,i] = factor * (-120 * q + 120 * q^2 - 50 * q^3) / hsmlij

            dWijdx[count,i] = factor * ((-120 + 120q - 50q^2) / hsmlij^2 * dxij)
            dWijdy[count,i] = factor * ((-120 + 120q - 50q^2) / hsmlij^2 * dyij)
            
        elseif 1 < q <= 2

            Wij[count,i]    = factor * ((3 - q)^5 - 6 * (2 - q)^5)
            dWijdr[count,i] = factor * (-5 * (3 - q)^4 + 30 * (2 - q)^4) / hsmlij
            dWijdx[count,i] = factor * (-5 * (3 - q)^4 + 30 * (2 - q)^4) / hsmlij * (dxij / Rij)
            dWijdy[count,i] = factor * (-5 * (3 - q)^4 + 30 * (2 - q)^4) / hsmlij * (dyij / Rij)
            
        elseif 2 < q <= 3

            Wij[count,i]    = factor * (3 - q)^5
            dWijdr[count,i] = factor * (-5 * (3 - q)^4) / hsmlij
            dWijdx[count,i] = factor * (-5 * (3 - q)^4) / hsmlij * (dxij / Rij)
            dWijdy[count,i] = factor * (-5 * (3 - q)^4) / hsmlij * (dyij / Rij)
            
        end

    end

    return nothing
    
end
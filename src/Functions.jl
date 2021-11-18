using CSV, DataFrames, Dates, PyCall

export JuliaToPandas
export PandasToJulia

"convert a Julia DataFrame to a Pandas DataFrame"
function JuliaToPandas(df::DataFrame)
    pd = pyimport("pandas")

    date_col_name = (eltype.(eachcol(df)) .|> string .== "Date") |> x-> names(df)[x] |> x-> x[1]
    CSV.write("tmp.csv", df)
    pd_df = pd.read_csv("tmp.csv", parse_dates=[date_col_name])
    run(`rm tmp.csv`)
    return pd_df
end

"convert a Pandas DataFrame to a Julia DataFrame"
function PandasToJulia(df::PyObject, missingstrings::Vector=["NA", "na", "nan", "NAN", "NaN", ".", ""])
    df.to_csv("tmp.csv", index=false)
    julia_df = CSV.File("tmp.csv", missingstrings=missingstrings) |> DataFrame
    run(`rm tmp.csv`)
    return julia_df
end

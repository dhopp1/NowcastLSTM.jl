using CSV, DataFrames, Dates, PyCall

export JuliaToPandas
export PandasToJulia
export LSTM
export train
export save_lstm
export load_lstm
export predict
export ragged_preds
export gen_news

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

"""
    Primary class of the library, used for transforming data, training the model, and making predictions.
    parameters:
        data : DataFrame
            n x m+1 dataframe
        target_variable : String
            name of the target var
        n_timesteps : Int
            how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
        fill_na_func : PyObject
            Python function to replace within-series NAs. Given a column, the function should return a scalar. E.g. to replace default mean with median, `LSTM(..., fill_na_func = pyimport("numpy").nanmedian)`
        fill_ragged_edges_func : PyObject
            Python function to replace NAs in ragged edges (data missing at end of series). Pass "ARMA" for ARMA filling. Not ARMA filling will be significantly slower as models have to be estimated for each variable to fill ragged edges.
        n_models : Int
            number of models to train and take the average of for more robust estimates
        train_episodes : Int
            number of epochs/episodes to train the model
        batch_size : Int
            number of observations per training batch
        decay : Float32
            learning rate decay
        n_hidden : Int
            number of hidden states in the network
        n_layers : Int
            number of LSTM layers in the network
        dropout : Float32
            dropout rate between the LSTM layers
        criterion : PyObject
            torch loss criterion, defaults to MAE. E.g. `LSTM(..., pyimport("torch").nn.L1Loss())`. Need parentheses at end of torch loss function.
        optimizer : PyObject
            torch optimizer, defaults to Adam. E.g. `LSTM(..., pyimport("torch").optim.Adam)`. Don't need parentheses at end of torch loss function.
        optimizer_parameters : Dict
            list of parameters for optimizer, including learning rate. E.g. Dict("lr" => 1e-2)
    returns: PyObject
        instantiated LSTM model
"""
function LSTM(;
    data::DataFrame = nothing,
    target_variable::String = nothing,
    n_timesteps::Int = nothing,
    fill_na_func::PyObject = pyimport("numpy").nanmean,
    fill_ragged_edges_func = pyimport("numpy").nanmean,
    n_models::Int = 1,
    train_episodes::Int = 200,
    batch_size::Int = 30,
    decay::Float64 = 0.98,
    n_hidden::Int = 20,
    n_layers::Int = 2,
    dropout::Float64 = 0.0,
    criterion::PyObject = pyimport("torch").nn.L1Loss(),
    optimizer::PyObject = pyimport("torch").optim.Adam,
    optimizer_parameters::Dict = Dict("lr" => 1e-2)
)::PyObject

    model = pyimport("nowcast_lstm.LSTM").LSTM(
        data = JuliaToPandas(data),
        target_variable = target_variable,
        n_timesteps = n_timesteps,
        fill_na_func = fill_na_func,
        fill_ragged_edges_func = fill_ragged_edges_func,
        n_models = n_models,
        train_episodes = train_episodes,
        batch_size = batch_size,
        decay = decay,
        n_hidden = n_hidden,
        n_layers = n_layers,
        dropout = dropout,
        criterion = criterion,
        optimizer = optimizer,
        optimizer_parameters = optimizer_parameters,
    )

    return model
end

"train an instantiated LSTM model. pass train(model; quiet=true). quiet=true will suppress printing of training loss at each epoch"
function train(model::PyObject; quiet::Bool = false)::PyObject
    model.train(quiet=quiet)
    return model
end

"get predictions from a trained model. pass predict(model, data; only\\_actuals\\_obs = true) to only get predictions where there is an actual value"
function predict(model::PyObject, data::DataFrame; only_actuals_obs::Bool = false)::DataFrame
    model.predict(JuliaToPandas(data), only_actuals_obs) |> PandasToJulia
end

"save a trained LSTM model to disk. pass save_lstm(model, path_to_save), filename must end in .pkl, e.g. '/Users/trained_model.pkl'"
function save_lstm(model::PyObject, filename::String)
    pyimport("dill").dump(model, py"open"(filename, mode="wb"))
end

"load a trained LSTM model. pass load_lstm(path_to_pkl)"
function load_lstm(filename::String)::PyObject
    model = pyimport("dill").load(py"open"(filename, "rb", -1))
    return model
end

"""
    Get predictions on artificial vintages
    parameters:
        model : PyObject
            trained LSTM model
        pub_lags : Vector
            list of periods back each input variable is set to missing. I.e. publication lag of the variable.
        lag : Int
            simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
        data : DataFrame
            dataframe to generate the ragged datasets on, if none will calculate on training data
        start_date : String
            String in "YYYY-MM-DD" format: start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
        end_date : String
            String in "YYYY-MM-DD" format: end date of generating ragged preds
    returns: DataFrame
        predictions on vintages
"""
function ragged_preds(
    model::PyObject,
    pub_lags::Vector{Int64},
    lag::Int64,
    data::DataFrame;
    start_date::String = "",
    end_date::String = "",
)::DataFrame
    if start_date == ""
        start_date = nothing
    end
    if end_date == ""
        end_date = nothing
    end
    model.ragged_preds(pub_lags, lag, JuliaToPandas(data), start_date, end_date) |> PandasToJulia
end

"""
    Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output
    parameters:
        model : PyObject
            trained LSTM model
        target_period : String
            String in "YYYY-MM-DD", target prediction date
        old_data : DataFrame
             dataframe of previous dataset
        new_data : DataFrame
             dataframe of new dataset
    returns: Dictionary
        "news": dataframe of news contribution of each column with updated data. scaled_news is news scaled to sum to actual prediction delta.
        "old_pred": prediction on the previous dataset
        "new_pred": prediction on the new dataset
        "holdout_discrepency": difference between the sum of news via the holdout method and the actual prediction delta
"""
function gen_news(model::PyObject, target_period::String, old_data::DataFrame, new_data::DataFrame)::Dict
    news = model.gen_news(target_period, JuliaToPandas(old_data), JuliaToPandas(new_data))
    julia_news = Dict()
    julia_news["news"] = news["news"] |> PandasToJulia
    julia_news["old_pred"] = news["old_pred"]
    julia_news["new_pred"] = news["new_pred"]
    julia_news["holdout_discrepency"] = news["holdout_discrepency"]
    return julia_news
end

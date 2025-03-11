open Iree_bindings

module Device =
  ( val Pjrt_bindings.make
          "/home/michel/part-ii-project/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    )

module Runtime = Runtime.Make (Device)
open Runtime

(* let benchmark_fn f size = *)
(*   let f = compile [] @@ fun Ir.Var.List.[] -> E (mean [0] @@ f [size]) in *)
(*   fun () -> *)
(*     let y = f [] in *)
(*     ignore @@ DeviceValue.to_host_value y *)

(* let uniform = benchmark_fn Random.uniform_f32 *)

(* let normal = benchmark_fn Random.normal_f32 *)

module Result = struct
  type t = {times: float list}

  let to_json {times} = `List (List.map (fun x -> `Float x) times)
end

let benchmark f n_warmup n_bench =
  for _ = 1 to n_warmup do
    ignore @@ f ()
  done ;
  let times =
    List.init n_bench (fun _ ->
        Core.Time_ns.(
          let start_time = now () in
          f () ;
          let end_time = now () in
          let elapsed_time = diff end_time start_time in
          let ns = Span.to_ns elapsed_time in
          Base.Float.to_float ns /. 1e6 ) )
  in
  Result.{times}

module StringMap = Map.Make (String)

module Grid = struct
  type t = {results: Result.t StringMap.t; label: string}

  let to_json {results; label} =
    `Assoc
      [ ("label", `String label)
      ; ( "results"
        , `Assoc
            ( StringMap.bindings results
            |> List.map (fun (name, result) -> (name, Result.to_json result)) )
        ) ]

  let save grid filename =
    let json = to_json grid in
    Yojson.Basic.to_file filename json
end

let benchmark_grid f label values n_warmup n_bench =
  let results =
    List.fold_left
      (fun acc value ->
        let f = f value in
        let times = benchmark f n_warmup n_bench in
        StringMap.add (string_of_int value) times acc )
      StringMap.empty values
  in
  Grid.{results; label}

let rec range start stop step =
  if start >= stop then [] else start :: range (start + step) stop step

(* let sizes = *)
(*   range (Base.Int.pow 2 26) (Base.Int.pow 2 29 + 1) (Base.Int.pow 2 26) *)

(* let () = *)
(*   let grid = benchmark_grid uniform "This implementation" sizes 100 1000 in *)
(*   Grid.save grid "uniform.json" ; *)
(*   let grid = benchmark_grid normal "This implementation" sizes 100 1000 in *)
(*   Grid.save grid "normal.json" *)

(* let mnist batch_size = *)
(*   let dataset = Mnist.load_images Train in *)
(*   let dataset = Dataset.shuffle dataset in *)
(*   let dataset = Dataset.batch_tensors batch_size dataset in *)
(*   let dataset = Dataset.repeat ~total:1_000_000 dataset in *)
(*   let dataset = Dataset.to_seq ~num_workers:8 dataset in *)
(*   let dataset = ref dataset in *)
(*   fun () -> *)
(*     match Seq.uncons !dataset with *)
(*     | Some (x, xs) -> *)
(*         dataset := xs ; *)
(*         ignore x *)
(*     | None -> *)
(*         failwith "End of dataset" *)

(* let sizes = range 64 (512 + 1) 64

let () =
  let grid = benchmark_grid mnist "MNIST" sizes 100 1000 in
  Grid.save grid "mnist.json"*)

let batch_sizes = range 64 (512 + 1) 64

let mnist batch_size =
  let dataset = Mnist.load_images Train in
  let dataset = Dataset.shuffle dataset in
  let dataset = Dataset.batch_tensors batch_size dataset in
  let dataset = Dataset.repeat ~total:10_000 dataset in
  let dataset =
    Dataset.map (fun x -> DeviceValue.of_host_value @@ E x) dataset
  in
  Dataset.to_seq ~num_workers:8 dataset

let one_sample batch_size =
  let dataset = Mnist.load_images Train in
  let dataset = Dataset.batch_tensors batch_size dataset in
  let dataset = Dataset.repeat ~total:1 dataset in
  let dataset = Dataset.to_seq dataset in
  match Seq.uncons dataset with
  | Some (x, _) ->
      x
  | None ->
      failwith "End of dataset"

let train_step_without_loading batch_size =
  let input_type = ([batch_size; 1; 784], Ir.Tensor.F32) in
  let param_type = Parameters.param_type (E input_type) Vae.train in
  let train_step =
    compile [param_type; E input_type]
    @@ fun [params; x] -> Parameters.to_fun (Vae.train x) params
  in
  let params =
    Parameters.initial (E input_type) Vae.train
    |> DeviceValue.of_host_value |> ref
  in
  let batch = one_sample batch_size in
  let batch = DeviceValue.of_host_value @@ E batch in
  fun () ->
    let [loss; params'] = train_step [!params; batch] in
    params := params' ;
    ignore @@ DeviceValue.to_host_value loss

let train_step_with_loading batch_size =
  let input_type = ([batch_size; 1; 784], Ir.Tensor.F32) in
  let param_type = Parameters.param_type (E input_type) Vae.train in
  let train_step =
    compile [param_type; E input_type]
    @@ fun [params; x] -> Parameters.to_fun (Vae.train x) params
  in
  let params =
    Parameters.initial (E input_type) Vae.train
    |> DeviceValue.of_host_value |> ref
  in
  let dataset = ref @@ mnist batch_size in
  fun () ->
    let batch, dataset' = Option.get @@ Seq.uncons !dataset in
    dataset := dataset' ;
    let [loss; new_params] = train_step [!params; batch] in
    params := new_params ;
    ignore @@ DeviceValue.to_host_value loss

let () =
  let grid =
    benchmark_grid train_step_without_loading "This implementation" batch_sizes
      100 1000
  in
  Grid.save grid "vae_without_loading.json" ;
  let grid =
    benchmark_grid train_step_with_loading "This implementation" batch_sizes 100
      1000
  in
  Grid.save grid "vae_with_loading.json"

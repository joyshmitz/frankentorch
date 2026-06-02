use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_dispatch::{SchemaRegistry, parse_schema_or_name, schema_dispatch_keyset_from_tags};

fn bench_schema_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_registry");
    let bases = [
        "add",
        "sub",
        "div",
        "mul",
        "matmul",
        "min",
        "max",
        "dot",
        "outer",
        "bmm",
        "atan2",
        "fmod",
        "remainder",
    ];
    let parsed = bases
        .iter()
        .copied()
        .cycle()
        .take(1024)
        .enumerate()
        .map(|(idx, base)| {
            parse_schema_or_name(&format!("{base}.bench_{idx}")).expect("bench schema parses")
        })
        .collect::<Vec<_>>();
    let keyset =
        schema_dispatch_keyset_from_tags(&["CPU", "AutogradCPU"]).expect("bench keyset parses");

    group.bench_function("register_1024", |b| {
        b.iter(|| {
            let mut registry = SchemaRegistry::new();
            for schema in &parsed {
                registry
                    .register(schema, keyset)
                    .expect("unique bench schema registers");
            }
            black_box(registry.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_schema_registry);
criterion_main!(benches);

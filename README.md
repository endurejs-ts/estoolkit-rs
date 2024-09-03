# estoolkit-rs

rust version of estoolkit (exclude function, number, promise, object, ...)

# Eng

[es-toolkit](https://github.com/toss/es-toolkit) is a high performance javascript library
this repository is es-toolkit that made of Rust.

# Kor

[es-toolkit](https://github.com/toss/es-toolkit)은 토스에서 만든 고성능 자바스크립트 라이브러리 입니다.
이 리포지토리는 es-toolkit을 러스트로 만들었습니다.

# Usage [사용법]

> 참고로 function, number, promise, object의 관련 기능들은 제작자의 한계로 구현하지 못했습니다. 추후 실력자의 기여로 구현 되었으면 좋겠네요.

## Array

### 1. Array chunk

```rust
// not import-export utility yet
// use estoolkit_rs::chunk;

fn main() {
    let arr = &[1, 2, 3, 4, 5, 6];
    let chunk_size = 2;

    let chunks = chunk(arr, chunk_size);
    println!("{:?}", chunks); // Output: [[1, 2], [3, 4], [5, 6]]

    // if size == 0
    let chunks_2 = chunk(arr, 0);
    println!("{:?}", chunks_2); // Output: []
}
```

### 2. Array uniq

```rust
// not import-export utility yet
// use estoolkit_rs::uniq;

fn main() {
    let arr = &[1, 2, 2, 3, 3, 4, 4, 5];
    let uniq_arr = uniq(arr);

    println!("{:?}", uniq_arr); // Output: [1, 2, 3, 4, 5]
}
```

[더보기] more at [here](https://github.com/endurejs-ts/estoolkit-rs)

## String

### 1. String camel case

```rust
// use estoolkit_rs::cameL_case;

fn main() {
    println!("{:?}", camel_case("hello world rust")); // Output: helloWorldRust
}
```

### 2. String kebab case

```rust
// use estoolkit_rs::kebab_case;

fn main() {
    println!("{:?}", kebab_case("helloWorld Rust")); // Output: hello-world-rust
}
```

[또한, 더보기] also, more at [here](https://github.com/endurejs-ts/estoolkit-rs)

<hr>

# Contributions [기여]
want contributions? start at [here]()